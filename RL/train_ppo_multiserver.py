# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from client_pensieve import Client
import random
import os
import numpy as np
import time
import platform
import glob
import sys
from torch.amp import autocast, GradScaler
import cProfile
import pstats
from trace_process.bandwidth_lstm import BandwidthLSTM

# Add after import statements
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Environment parameters
M_IN_K = 1000
BITRATES = [350, 600, 1000, 2000, 3000]
S_LEN = 7
# A_DIM = 6
# Original action space: 5 bitrates + 1 miss = 6
# New action space: (5 bitrates + 1 miss) * 3 edge servers = 18
A_DIM = 18  # 6 * 3
throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455

# System parameters
os.environ["OMP_NUM_THREADS"] = "1"
if platform.system() == "Linux":
    MAX_EP = 30000
    PRINTFLAG = False
else:
    MAX_EP = 40
    PRINTFLAG = True

# PPO algorithm parameters
GAMMA = 0.99                # discount factor
LAMBDA = 0.95              # GAE parameter
VALUE_COEF = 1.0           # value function loss coefficient
ENTROPY_COEF = 0.02        # entropy regularization coefficient
CLIP_EPSILON = 0.1         # PPO clipping parameter
MAX_GRAD_NORM = 0.5        # gradient clipping threshold

# Training parameters
BATCH_SIZE = 1024          # decrease batch size
MINI_BATCH_SIZE = 256      # decrease accordingly
PPO_EPOCHS = 4             # decrease update times
LR = 1e-4                  # initial learning rate
NUM_WORKERS = 4            # number of workers

# Early stopping parameters
EARLY_STOP_REWARD = 400    # increase target reward threshold
PATIENCE = 100             # increase patience value

# Add new hyperparameters
REWARD_SCALE = 1.0  # increase reward scaling factor from 0.1 to 1.0
MIN_LR = 1e-5      # minimum learning rate
LR_DECAY_RATE = 0.995  # learning rate decay rate
LR_DECAY_STEPS = 1000  # learning rate decay steps
MAX_GRAD_NORM = 0.5    # decrease gradient clipping threshold
MIN_EPISODES = 5000    # minimum training rounds





def takeSecond(elem):
    return elem[1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # actor network
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.ln_1_a = nn.LayerNorm(200)
        self.linear_2_a = nn.Linear(200, 100)
        self.ln_2_a = nn.LayerNorm(100)
        self.output_a = nn.Linear(100, A_DIM)
        
        # critic network
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.ln_1_c = nn.LayerNorm(200)
        self.linear_2_c = nn.Linear(200, 100)
        self.ln_2_c = nn.LayerNorm(100)
        self.output_c = nn.Linear(100, 1)
        
        # initialize weights
        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                 self.linear_1_c, self.linear_2_c, self.output_c])
        
        self.distribution = torch.distributions.Categorical
        
        # move model to GPU
        self.to(device)

    def forward(self, x):
        # ensure input is on GPU
        x = x.to(device)
        
        # actor forward pass
        a = F.relu(self.ln_1_a(self.linear_1_a(x)))
        a = F.relu(self.ln_2_a(self.linear_2_a(a)))
        logits = self.output_a(a)
        
        # critic forward pass
        c = F.relu(self.ln_1_c(self.linear_1_c(x)))
        c = F.relu(self.ln_2_c(self.linear_2_c(c)))
        values = self.output_c(c)
        
        return logits, values

    # def choose_action(self, mask, state):
    #     self.eval()
    #     # move input to GPU and adjust dimensions
    #     state = torch.FloatTensor(state).unsqueeze(0).to(device)  # add batch dimension
    #     mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)  # add batch dimension
        
    #     with torch.no_grad():
    #         logits, value = self.forward(state)
    #         masked_logits = logits.clone()
    #         masked_logits[~mask] = float('-inf')  # now dimensions match
    #         masked_logits = masked_logits - masked_logits.max()
    #         probs = F.softmax(masked_logits, dim=-1)
            
    #         if torch.isnan(probs).any():
    #             print("Warning: NaN values in probabilities:", probs)
            
    #         dist = self.distribution(probs)
    #         action = dist.sample()
    #         action_log_prob = dist.log_prob(action)
            
    #     return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()
    def choose_action(self, mask, state, deterministic=False):  # add deterministic parameter
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            
            logits, value = self.forward(state)
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            
            # use softmax to get base probabilities
            probs = F.softmax(masked_logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                # if exploration is needed, add small noise to logits
                noise = torch.randn_like(masked_logits) * 0.1
                noisy_logits = masked_logits + noise
                # re-softmax to ensure valid probability distribution
                noisy_probs = F.softmax(noisy_logits, dim=-1)
                dist = self.distribution(noisy_probs)
                action = dist.sample()
            
            # use original probabilities to calculate log prob
            action_log_prob = torch.log(probs[0, action])
            
            # print action probability distribution and selected action
            # print(f"\nAction probabilities: {probs[0].cpu().numpy()}")
            # print(f"Selected action: {action.item()}, Probability: {probs[0][action].item():.4f}")
            
            return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()

    def compute_loss(self, states, actions, old_log_probs, returns, advantages, masks):
        # convert all inputs to tensors and move to GPU
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        masks = torch.tensor(masks, dtype=torch.bool).to(device)
        
        # compute new action probabilities and values
        logits, values = self.forward(states)
        # ... rest of loss computation code remains the same ...

    @torch.no_grad()  # disable gradient calculation
    def evaluate(self, states):
        return self.policy(states)

    def get_value(self, state):
        # remove v_wrap, directly use state
        if not state.dim():  # if scalar
            state = state.unsqueeze(0)
        if len(state.shape) == 1:  # if one-dimensional tensor
            state = state.unsqueeze(0)
        _, value = self.forward(state)
        return value.item()


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, transition):
        """Store transition data"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of data"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.returns = []
        self.advantages = []
        
        # add experience replay buffer
        self.replay_buffer = ReplayBuffer()
        
    def store(self, state, action, log_prob, value, reward, mask):
        # store transition
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        # also store to experience replay buffer
        self.replay_buffer.push({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'mask': mask
        })
        
    def compute_returns(self, last_value, gamma, lambda_):
        # compute returns and advantage function
        gae = 0
        returns = []
        advantages = []  # add advantage function list
        values = self.values + [last_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)  # store advantage function
            returns.insert(0, gae + values[step])
            
        self.returns = returns
        self.advantages = advantages  # store computed advantage function
        return returns

    def get_batches(self, batch_size):
        # generate mini-batch
        indices = np.random.permutation(len(self.states))
        
        # calculate the proportion of replay data to use
        replay_ratio = 0.3  # adjustable parameter
        replay_size = int(batch_size * replay_ratio)
        current_size = batch_size - replay_size

        for start in range(0, len(self.states), batch_size):
            end = start + current_size
            if end > len(self.states):
                break
                
            batch_indices = indices[start:end]
            
            # get current trajectory data
            current_batch = {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'log_probs': [self.log_probs[i] for i in batch_indices],
                'returns': [self.returns[i] for i in batch_indices],
                'advantages': [self.advantages[i] for i in batch_indices],
                'masks': [self.masks[i] for i in batch_indices]
            }
            
            # sample from experience replay buffer
            if len(self.replay_buffer) > replay_size:
                replay_samples = self.replay_buffer.sample(replay_size)
                
                # merge replay data
                for replay in replay_samples:
                    current_batch['states'].append(replay['state'])
                    current_batch['actions'].append(replay['action'])
                    current_batch['log_probs'].append(replay['log_prob'])
                    current_batch['masks'].append(replay['mask'])
                    # use approximate returns and advantages for replay data
                    current_batch['returns'].append(replay['reward'])  # simplified handling
                    current_batch['advantages'].append(0.0)  # simplified handling
            
            # convert to numpy arrays
            states = np.array(current_batch['states'])
            actions = np.array(current_batch['actions'])
            log_probs = np.array(current_batch['log_probs'])
            returns = np.array(current_batch['returns'])
            advantages = np.array(current_batch['advantages'])
            
            yield {
                'states': torch.FloatTensor(states),
                'actions': torch.LongTensor(actions),
                'log_probs': torch.FloatTensor(log_probs),
                'returns': torch.FloatTensor(returns).view(-1),
                'advantages': torch.FloatTensor(advantages).view(-1),
                'masks': current_batch['masks']
            }

    def clear(self):
        # complete clear method
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.returns.clear()
        self.advantages.clear()


class RunningMeanStd: 
    # state normalization
    def __init__(self):
        # initialize state normalization
        self.mean = np.zeros(S_LEN)
        self.var = np.ones(S_LEN)
        self.count = 1e-4

    def update(self, x):
        # update state normalization statistics
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count)
        self.count += batch_count


def format_time(seconds):
    """Convert seconds to a readable time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

def clean_old_models(model_dir, keep_latest=5):
    """Keep the latest 5 model files"""
    try:
        model_files = []
        for f in os.listdir(f"{model_dir}/model"):
            if f.startswith('model_') and f.endswith('.pth'):
                try:
                    episode_num = int(f.split('_')[1].split('.')[0])
                    model_files.append((f, episode_num))
                except:
                    continue
        
        model_files.sort(key=lambda x: x[1])
        
        if len(model_files) > keep_latest:
            for f, _ in model_files[:-keep_latest]:
                file_path = f"{model_dir}/model/{f}"
                os.remove(file_path)
            print(f"\nKeep the latest {keep_latest} models:")
            for f, _ in model_files[-keep_latest:]:
                print(f"- {f}")
    except Exception as e:
        print(f"Error while cleaning old models: {e}")


class Worker:
    def __init__(self, model_dir):
        # initialization
        self.model_dir = model_dir
        self.policy = Net().to(device)  # policy network
        # ensure model parameters are float32
        self.policy = self.policy.float()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        
        # add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',           # as we want to maximize reward
            factor=LR_DECAY_RATE, # learning rate decay factor
            patience=10,          # number of epochs with no improvement before decay
            min_lr=MIN_LR,        # minimum learning rate
            verbose=True          # print learning rate changes
        )
        
        # use new Client class
        self.client = Client("train", model_dir)
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]
        self.busyTraceL = os.listdir("../data/trace/busy/2")
        self.bwType = 3
        self.rttDict = {}
        get_rttDict(self.rttDict)
        self.rewards = []
        
        # initialize current state
        self.current_state = np.zeros(S_LEN)  # initialize state with numpy array
        self.segmentNum = 0
        
        # add reward processing parameters
        self.reward_scale = 100.0  # reward scaling factor
        self.reward_clip_min = -10.0  # reward clipping lower bound
        self.reward_clip_max = 10.0   # reward clipping upper bound
        
        # initialize bandwidth LSTM model
        self.bandwidth_lstms = {
            "edge1": BandwidthLSTM().to(device),
            "edge2": BandwidthLSTM().to(device),
            "edge3": BandwidthLSTM().to(device)
        }
        
        self.rewards_history = []  # add rewards_history list to record rewards
        
    def process_reward(self, reward):
        """Reward processing function - remove normalization"""
        # only perform basic clipping, keep original reward signal
        clipped_reward = np.clip(reward, -10, 10)
        return clipped_reward
    
    def get_video(self):
        """Get video file, use probability selection"""
        while True:
            video_random = random.random()
            videoName = ""
            for i in range(len(self.videoList)):
                if video_random < self.videoList[i][1]:
                    videoName = self.videoList[i - 1][0]
                    break
            if videoName == "":
                videoName = self.videoList[-1][0]
            else:
                break
        return videoName

    def get_busyTrace(self):
        """Get busy trace data"""
        fileName = random.choice(self.busyTraceL)
        return np.loadtxt("../data/trace/busy/2/" + fileName).flatten().tolist()

    def getBandwidthFile(self):
        """Get bandwidth files for each edge server"""
        bandwidth_files = {}
        rtt_values = {}
        
        # edge1 to client bandwidth (variable: FCC, HSDPA, mine)
        self.bwType = random.randint(1, 3)
        if self.bwType == 1:
            edge1_dir = "../data/bandwidth/train/FCC"
        elif self.bwType == 2:
            edge1_dir = "../data/bandwidth/train/HSDPA"
        else:
            edge1_dir = "../data/bandwidth/train/mine"
        
        # edge2 and edge3 use 5G Netflix data
        edge2_dir = "../data/bandwidth/train/5G_Neflix_static"
        edge3_dir = "../data/bandwidth/train/5G_Neflix_static"
        
        try:
            # handle edge1 bandwidth files
            if self.bwType == 3:
                ip_dirs = os.listdir(edge1_dir)
                ip = random.choice(ip_dirs)
                if ip not in self.rttDict:
                    print("no this ip:", ip, "in the bandwidth directory")
                    edge1_dir = "../data/bandwidth/train/HSDPA"
                    bandwidth_files["edge1"] = random.choice(os.listdir(edge1_dir))
                else:
                    rtt_values["edge1"] = self.rttDict[ip]
                    bw_files = os.listdir(os.path.join(edge1_dir, ip))
                    bandwidth_files["edge1"] = os.path.join(edge1_dir, ip, random.choice(bw_files))
            else:
                bw_files = os.listdir(edge1_dir)
                bandwidth_files["edge1"] = os.path.join(edge1_dir, random.choice(bw_files))
                rtt_values["edge1"] = self.rttDict.get(self.bwType, -1)
            
            # handle edge2 bandwidth files (5G Netflix)
            edge2_files = os.listdir(edge2_dir)
            bandwidth_files["edge2"] = os.path.join(edge2_dir, random.choice(edge2_files))
            rtt_values["edge2"] = 20  # typical RTT value for 5G networks
            
            # handle edge3 bandwidth files (5G Netflix)
            edge3_files = os.listdir(edge3_dir)
            bandwidth_files["edge3"] = os.path.join(edge3_dir, random.choice(edge3_files))
            rtt_values["edge3"] = 20  # typical RTT value for 5G networks
            
        except Exception as e:
            print(f"Error getting bandwidth files: {e}")
            # use default bandwidth files
            for server in ["edge1", "edge2", "edge3"]:
                bandwidth_files[server] = self.create_default_bandwidth_file()
                rtt_values[server] = -1
        
        return bandwidth_files, rtt_values

    def create_default_bandwidth_file(self):
        """Create default bandwidth data file"""
        # create a temporary file to store default bandwidth data
        temp_dir = "../data/bandwidth/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, "default_bandwidth.txt")
        with open(file_path, 'w') as f:
            # generate 100 time points of bandwidth data
            for i in range(100):
                # timestamp bandwidth(bps)
                f.write(f"{i} 1000000\n")
        
        return file_path

    def step(self, action):
        """Execute action and obtain reward"""
        # get server index and bitrate index
        server_idx = action // 6  # 0,1,2 correspond to three servers
        bitrate_idx = action % 6  # 0-5 correspond to bitrate options
        
        # determine if cache hit:
        # 1. bitrate_idx is not 5 (not a miss action)
        # 2. and the action is 1 in the mask (the bitrate is in cache)
        hitFlag = bitrate_idx != 5
        
        server_name = f"edge{server_idx + 1}"
        
        # get current busy
        busy = self.busyList[self.segmentNum % len(self.busyList)]
        
        # execute action and directly use reward from client.py
        reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, \
        reward, reqBI, done, segmentNum = self.client.run(bitrate_idx, busy, hitFlag, server_name)
        
        # process reward
        processed_reward = self.process_reward(reward)
        
        # update state correctly handle last_server_id
        last_server_idx = self.current_state[5] * 2  # restore unnormalized value
        self.current_state = [
            reqBitrate / BITRATES[-1],
            lastBitrate / BITRATES[-1],
            (buffer/1000 - 30) / 10,
            (hThroughput - throughput_mean) / throughput_std,
            (mThroughput - throughput_mean) / throughput_std,
            server_idx / 2,  # current server ID
            last_server_idx / 3  # previous server ID (use unnormalized value)
        ]
        
        self.segmentNum = segmentNum
        return processed_reward, done

    def init_state(self):
        """Initialize state"""
        return {
            'reqBitrate': 0,
            'lastBitrate': 0,
            'bufferSize': 0,
            'hThroughput': 0,
            'mThroughput': 0
        }

    def get_state(self):
        # get current server ID
        if self.client.current_server is None:
            server_id = 0  # default to edge1
        else:
            server_id = int(self.client.current_server.split('edge')[1]) - 1
        
        # get previous server ID
        if self.client.last_server is None:
            last_server_id = -1
        else:
            try:
                last_server_id = int(self.client.last_server.split('edge')[1]) - 1
            except (AttributeError, IndexError, ValueError):
                last_server_id = -1
        
        state = np.array([
            self.current_state[0],  # reqBitrate
            self.current_state[1],  # lastBitrate
            self.current_state[2],  # bufferSize
            self.current_state[3],  # hThroughput
            self.current_state[4],  # mThroughput
            self.current_state[5],  # server_id
            self.current_state[6],  # last_server_id
        ])
        
        return state
     

    def update_policy(self, states, actions, rewards):
        """Update policy network"""
        # ensure all inputs are float32 type
        states = torch.stack([s.float() for s in states]).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)  # action should be long type
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # compute advantage values
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()
            advantages = rewards - values
            # standardize advantage values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_rewards = rewards[mb_indices]
                
                logits, values = self.policy(mb_states)
                new_probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(mb_actions)
                
                ratio = torch.exp(new_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                
                # compute policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # compute value function loss
                value_loss = F.mse_loss(values.squeeze(), mb_rewards)
                
                # compute entropy reward
                entropy = dist.entropy().mean()
                
                # total loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # use smaller gradient clipping threshold
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # update learning rate
        self.scheduler.step()
        
        return total_policy_loss / PPO_EPOCHS, total_value_loss / PPO_EPOCHS

    def process_bandwidth(self, hThroughput, server_name):
        """Use LSTM to process bandwidth data"""
        processed_throughput = self.bandwidth_lstms[server_name].process_throughput(hThroughput)
        return (processed_throughput - throughput_mean) / throughput_std

    def get_action_mask(self):
        """Generate action mask for each server"""
        mask = [1] * A_DIM  # A_DIM = 18
        
        # generate mask for each server
        for server_idx in range(3):  # 3 servers
            base_idx = server_idx * 6  # starting index for each server
            
            # randomly decide how many versions to cache
            randmCachedBICount = random.randint(1, 5)
            BI = [0, 1, 2, 3, 4]  # 5 bitrate versions
            randomCachedBI = random.sample(BI, randmCachedBICount)
            
            # set mask for corresponding versions
            for bIndex in range(5):
                action_idx = base_idx + bIndex
                if bIndex not in randomCachedBI:
                    mask[action_idx] = 0
            
            # ensure miss action is always available
            mask[base_idx + 5] = 1
        
        return mask

    def run(self):
        """Main training loop"""
        print("Starting training...")
        episode = 0
        best_reward = float('-inf')
        patience_counter = 0
        running_reward = 0
        last_server_id = -1  # initialize last_server_id
        
        while episode < MAX_EP:
            # initialize environment
            self.videoName = self.get_video()
            self.bandwidth_file, self.rtt = self.getBandwidthFile()
            self.busyList = self.get_busyTrace()
            self.segmentNum = 0
            
            # generate action mask once for this episode and save
            self.current_mask = self.get_action_mask()
            
            # initialize client
            reqBI = self.client.init(self.videoName, self.bandwidth_file, self.rtt, self.bwType)
            
            # initialize state
            self.current_state = [
                reqBI / BITRATES[-1],  # normalized requested bitrate
                -1 / BITRATES[-1],     # normalized last bitrate
                0,                     # normalized buffer size
                0,                     # normalized historical throughput
                0,                     # normalized predicted throughput
                0,                     # normalized current server_id
                -1/3                   # normalized previous server_id
            ]
            
            # initialize trajectory storage
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            episode_reward = 0
            episode_steps = 0
            
            # collect trajectory
            while True:
                # get current state
                state = self.get_state()
                
                # choose action
                # exploration_rate = max(0.1, 1.0 - episode / MAX_EP)  # decrease exploration over time
                # deterministic = np.random.random() > exploration_rate
                deterministic = True

                action, log_prob, value = self.policy.choose_action(self.current_mask, state, deterministic)

                # execute action and obtain reward
                # reward, done = self.step(action)
                
                # print current action's reward
                # print(f"Action {action} reward: {reward:.4f}")

                # handle action
                server_idx = action // 6  # determine which edge server to choose
                bitrate_action = action % 6  # determine bitrate or miss
                
                # get corresponding server name
                server_name = f"edge{server_idx + 1}"
                
                # save current server_id before executing action
                last_server_id = server_idx
                
                # execute action
                busy = self.busyList[self.segmentNum % len(self.busyList)]
                hitFlag = True if bitrate_action != 5 else False
                
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = \
                    self.client.run(bitrate_action, busy, hitFlag, server_name)
                reward = reward / 5
                # print detailed training status
                # print(f"Episode {episode:5d} | "
                #       f"Step {len(states):4d} | "
                #       f"Seg {segNum:4d} | "
                #       f"ReqBR: {reqBitrate:5d} | "
                #       f"LastBR: {lastBitrate:5d} | "
                #       f"Buffer: {buffer/1000:6.2f}s | "
                #       f"HTput: {hThroughput/1000000:6.2f}Mbps | "
                #       f"MTput: {mThroughput/1000000:6.2f}Mbps | "
                #       f"Reward: {reward:6.2f} | "
                #       f"ReqBI: {reqBI} | "
                #       f"Done: {done} | "
                #       f"Server: {server_name}")
                
                # use LSTM to process bandwidth
                processed_throughput = self.process_bandwidth(hThroughput, server_name)
                
                # update state
                self.current_state = [
                    reqBitrate / BITRATES[-1],
                    lastBitrate / BITRATES[-1],
                    (buffer/1000 - 30) / 10,
                    # processed_throughput,  # use LSTM processed bandwidth
                    (hThroughput - throughput_mean) / throughput_std,
                    (mThroughput - throughput_mean) / throughput_std,
                    server_idx / 2,  # current server ID
                    (last_server_id + 1) / 3  # previous server ID
                ]
                
                # store trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                
                # accumulate episode reward
                episode_reward += reward
                episode_steps += 1
                
                # update state
                self.segmentNum = segNum
                
                # check if episode is done
                if self.segmentNum >= self.client.segmentCount:
                    last_server_id = -1
                    break
                
                # check if batch size is reached
                if len(states) >= BATCH_SIZE:
                    break
            
            
            # update running average reward (use smaller update rate for stability)
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            
            # record training information
            if episode % 10 == 0:
                # use running_reward to decide if learning rate decay is needed
                self.scheduler.step(running_reward)
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Episode {episode}, Running Reward: {running_reward:.2f}, LR: {lr:.6f}")
                
                # save best model
                if running_reward > best_reward:
                    best_reward = running_reward
                    if not PRINTFLAG:
                        torch.save(self.policy.state_dict(), 
                                 f"{self.model_dir}/model/best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # save model periodically
            if episode % 100 == 0 and not PRINTFLAG:
                torch.save(self.policy.state_dict(), 
                         f"{self.model_dir}/model/model_{episode}.pth")
            
            
            # early stopping check (only effective after minimum training rounds)
            if episode >= MIN_EPISODES:
                if patience_counter >= PATIENCE and running_reward > EARLY_STOP_REWARD:
                    print(f"Early stopping at episode {episode}")
                    break

            episode += 1
        
        print("Training completed!")
        
        # save final model
        if not PRINTFLAG:
            torch.save(self.policy.state_dict(), 
                     f"{self.model_dir}/model/final_model.pth")


def main():
    # create result saving directory
    if not PRINTFLAG:
        time_local = time.localtime(int(time.time()))
        dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
        model_dir = f"../data/RL_model/{dt}"
        os.makedirs(f"{model_dir}/model", exist_ok=True)
        
        # redirect standard output and error to log file
        log_file = f"{model_dir}/training.log"
        sys.stdout = open(log_file, 'w', buffering=1)
        sys.stderr = sys.stdout
    
    # create worker and train
    worker = Worker(model_dir)
    worker.run()

    # save training results
    if not PRINTFLAG:
        plt.figure(figsize=(10, 6))
        # plot original reward curve
        plt.plot(worker.rewards_history, 
                alpha=0.3, 
                color='blue', 
                label='Episode Reward')
        
        # plot moving average line
        window_size = 10
        moving_avg = np.convolve(
            worker.rewards_history, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        plt.plot(range(window_size-1, len(worker.rewards_history)), 
                moving_avg, 
                color='red', 
                label=f'{window_size}-Episode Moving Average')
        
        # set chart properties
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # save chart
        plt.savefig(f"{model_dir}/training_curve.pdf")
        plt.close()

if __name__ == "__main__":
    main()