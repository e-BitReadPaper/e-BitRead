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

# 在导入语句后添加
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 环境参数
M_IN_K = 1000
BITRATES = [350, 600, 1000, 2000, 3000]
S_LEN = 7
# A_DIM = 6
# 原来的动作空间: 5个码率 + 1个miss = 6
# 新的动作空间: (5个码率 + 1个miss) * 3个edge server = 18
A_DIM = 18  # 6 * 3
throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455

# 系统参数
os.environ["OMP_NUM_THREADS"] = "1"
if platform.system() == "Linux":
    MAX_EP = 30000
    PRINTFLAG = False
else:
    MAX_EP = 40
    PRINTFLAG = True

# PPO算法参数
GAMMA = 0.99                # 折扣因子
LAMBDA = 0.95              # GAE参数
VALUE_COEF = 1.0           # 值函数损失系数
ENTROPY_COEF = 0.02        # 熵正则化系数
CLIP_EPSILON = 0.1         # PPO裁剪参数
MAX_GRAD_NORM = 0.5        # 梯度裁剪阈值

# 训练参数
BATCH_SIZE = 1024          # 减小批量
MINI_BATCH_SIZE = 256      # 相应减小
PPO_EPOCHS = 4             # 减少更新次数
LR = 1e-4                  # 初始学习率
NUM_WORKERS = 4            # worker数量

# 早停参数
EARLY_STOP_REWARD = 400    # 提高目标奖励阈值
PATIENCE = 100             # 增加耐心值

# 添加新的超参数
REWARD_SCALE = 1.0  # 改大奖励缩放系数，从0.1增加到1.0
MIN_LR = 1e-5      # 最小学习率
LR_DECAY_RATE = 0.995  # 学习率衰减率
LR_DECAY_STEPS = 1000  # 学习率衰减步长
MAX_GRAD_NORM = 0.5    # 降低梯度裁剪阈值
MIN_EPISODES = 5000    # 最小训练轮数





def takeSecond(elem):
    return elem[1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # actor网络
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.ln_1_a = nn.LayerNorm(200)
        self.linear_2_a = nn.Linear(200, 100)
        self.ln_2_a = nn.LayerNorm(100)
        self.output_a = nn.Linear(100, A_DIM)
        
        # critic网络
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.ln_1_c = nn.LayerNorm(200)
        self.linear_2_c = nn.Linear(200, 100)
        self.ln_2_c = nn.LayerNorm(100)
        self.output_c = nn.Linear(100, 1)
        
        # 初始化权重
        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                 self.linear_1_c, self.linear_2_c, self.output_c])
        
        self.distribution = torch.distributions.Categorical
        
        # 将模型移到GPU
        self.to(device)

    def forward(self, x):
        # 确保输入在GPU上
        x = x.to(device)
        
        # actor前向传播
        a = F.relu(self.ln_1_a(self.linear_1_a(x)))
        a = F.relu(self.ln_2_a(self.linear_2_a(a)))
        logits = self.output_a(a)
        
        # critic前向传播
        c = F.relu(self.ln_1_c(self.linear_1_c(x)))
        c = F.relu(self.ln_2_c(self.linear_2_c(c)))
        values = self.output_c(c)
        
        return logits, values

    # def choose_action(self, mask, state):
    #     self.eval()
    #     # 将输入移到GPU，并调整维度
    #     state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 添加 batch 维度
    #     mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)  # 添加 batch 维度
        
    #     with torch.no_grad():
    #         logits, value = self.forward(state)
    #         masked_logits = logits.clone()
    #         masked_logits[~mask] = float('-inf')  # 现在维度匹配了
    #         masked_logits = masked_logits - masked_logits.max()
    #         probs = F.softmax(masked_logits, dim=-1)
            
    #         if torch.isnan(probs).any():
    #             print("Warning: NaN values in probabilities:", probs)
            
    #         dist = self.distribution(probs)
    #         action = dist.sample()
    #         action_log_prob = dist.log_prob(action)
            
    #     return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()
    def choose_action(self, mask, state, deterministic=False):  # 添加deterministic参数
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            
            logits, value = self.forward(state)
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            
            # 使用softmax获取基础概率
            probs = F.softmax(masked_logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                # 如果需要探索，添加较小的噪声到logits
                noise = torch.randn_like(masked_logits) * 0.1
                noisy_logits = masked_logits + noise
                # 重新进行softmax确保得到有效的概率分布
                noisy_probs = F.softmax(noisy_logits, dim=-1)
                dist = self.distribution(noisy_probs)
                action = dist.sample()
            
            # 使用原始概率计算log prob
            action_log_prob = torch.log(probs[0, action])
            
            # 打印动作概率分布和选择的动作
            # print(f"\nAction probabilities: {probs[0].cpu().numpy()}")
            # print(f"Selected action: {action.item()}, Probability: {probs[0][action].item():.4f}")
            
            return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()

    def compute_loss(self, states, actions, old_log_probs, returns, advantages, masks):
        # 将所有输入转换为张量并移到GPU
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        masks = torch.tensor(masks, dtype=torch.bool).to(device)
        
        # 计算新的动作概率和值
        logits, values = self.forward(states)
        # ... 其余损失计算代码保持不变 ...

    @torch.no_grad()  # 禁用梯度计算
    def evaluate(self, states):
        return self.policy(states)

    def get_value(self, state):
        # 移除 v_wrap，直接使用 state
        if not state.dim():  # 如果是标量
            state = state.unsqueeze(0)
        if len(state.shape) == 1:  # 如果是一维张量
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
        """存储转换数据"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """采样一个批次的数据"""
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
        
        # 添加经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
    def store(self, state, action, log_prob, value, reward, mask):
        # 存储transition
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        # 同时存储到经验回放缓冲区
        self.replay_buffer.push({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'mask': mask
        })
        
    def compute_returns(self, last_value, gamma, lambda_):
        # 计算回报和优势函数
        gae = 0
        returns = []
        advantages = []  # 添加优势函数列表
        values = self.values + [last_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)  # 存储优势函数
            returns.insert(0, gae + values[step])
            
        self.returns = returns
        self.advantages = advantages  # 存储计算的优势函数
        return returns

    def get_batches(self, batch_size):
        # 生成mini-batch
        indices = np.random.permutation(len(self.states))
        
        # 计算要使用的回放数据比例
        replay_ratio = 0.3  # 可调整的参数
        replay_size = int(batch_size * replay_ratio)
        current_size = batch_size - replay_size

        for start in range(0, len(self.states), batch_size):
            end = start + current_size
            if end > len(self.states):
                break
                
            batch_indices = indices[start:end]
            
            # 获取当前轨迹数据
            current_batch = {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'log_probs': [self.log_probs[i] for i in batch_indices],
                'returns': [self.returns[i] for i in batch_indices],
                'advantages': [self.advantages[i] for i in batch_indices],
                'masks': [self.masks[i] for i in batch_indices]
            }
            
            # 从经验回放缓冲区采样
            if len(self.replay_buffer) > replay_size:
                replay_samples = self.replay_buffer.sample(replay_size)
                
                # 合并回放数据
                for replay in replay_samples:
                    current_batch['states'].append(replay['state'])
                    current_batch['actions'].append(replay['action'])
                    current_batch['log_probs'].append(replay['log_prob'])
                    current_batch['masks'].append(replay['mask'])
                    # 为回放数据使用近似的returns和advantages
                    current_batch['returns'].append(replay['reward'])  # 简化处理
                    current_batch['advantages'].append(0.0)  # 简化处理
            
            # 转换为numpy数组
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
        # 完善clear方法
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.returns.clear()
        self.advantages.clear()


class RunningMeanStd: 
    # 状态标准化
    def __init__(self):
        # 初始化状态标准化
        self.mean = np.zeros(S_LEN)
        self.var = np.ones(S_LEN)
        self.count = 1e-4

    def update(self, x):
        # 更新状态标准化统计
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
    """将秒数转换为可读的时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

def clean_old_models(model_dir, keep_latest=5):
    """保留最新的5个模型文件"""
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
            print(f"\n保留最新的{keep_latest}个模型:")
            for f, _ in model_files[-keep_latest:]:
                print(f"- {f}")
    except Exception as e:
        print(f"清理旧模型时出错: {e}")


class Worker:
    def __init__(self, model_dir):
        # 初始化
        self.model_dir = model_dir
        self.policy = Net().to(device)  # 策略网络
        # 确保模型参数为 float32
        self.policy = self.policy.float()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',           # 因为我们要最大化奖励
            factor=LR_DECAY_RATE, # 学习率衰减因子
            patience=10,          # 等待多少个epoch无改善后才衰减
            min_lr=MIN_LR,        # 最小学习率
            verbose=True          # 打印学习率变化信息
        )
        
        # 使用新的Client类
        self.client = Client("train", model_dir)
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]
        self.busyTraceL = os.listdir("../data/trace/busy/2")
        self.bwType = 3
        self.rttDict = {}
        get_rttDict(self.rttDict)
        self.rewards = []
        
        # 初始化当前状态
        self.current_state = np.zeros(S_LEN)  # 使用numpy数组初始化状态
        self.segmentNum = 0
        
        # 添加奖励处理的参数
        self.reward_scale = 100.0  # 奖励缩放因子
        self.reward_clip_min = -10.0  # 奖励裁剪下限
        self.reward_clip_max = 10.0   # 奖励裁剪上限
        
        # 初始化带宽LSTM模型
        self.bandwidth_lstms = {
            "edge1": BandwidthLSTM().to(device),
            "edge2": BandwidthLSTM().to(device),
            "edge3": BandwidthLSTM().to(device)
        }
        
        self.rewards_history = []  # 添加rewards_history列表来记录奖励
        
    def process_reward(self, reward):
        """奖励处理函数 - 移除归一化"""
        # 只进行基本的裁剪，保持原始奖励信号
        clipped_reward = np.clip(reward, -10, 10)
        return clipped_reward
    
    def get_video(self):
        """获取视频文件,使用概率选择"""
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
        """获取忙碌度跟踪数据"""
        fileName = random.choice(self.busyTraceL)
        return np.loadtxt("../data/trace/busy/2/" + fileName).flatten().tolist()

    def getBandwidthFile(self):
        """获取每个edge server的带宽文件"""
        bandwidth_files = {}
        rtt_values = {}
        
        # edge1到client的带宽（可变：FCC, HSDPA, mine）
        self.bwType = random.randint(1, 3)
        if self.bwType == 1:
            edge1_dir = "../data/bandwidth/train/FCC"
        elif self.bwType == 2:
            edge1_dir = "../data/bandwidth/train/HSDPA"
        else:
            edge1_dir = "../data/bandwidth/train/mine"
        
        # edge2和edge3使用5G Netflix数据
        edge2_dir = "../data/bandwidth/train/5G_Neflix_static"
        edge3_dir = "../data/bandwidth/train/5G_Neflix_static"
        
        try:
            # 处理edge1的带宽文件
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
            
            # 处理edge2的带宽文件（5G Netflix）
            edge2_files = os.listdir(edge2_dir)
            bandwidth_files["edge2"] = os.path.join(edge2_dir, random.choice(edge2_files))
            rtt_values["edge2"] = 20  # 5G网络的典型RTT值
            
            # 处理edge3的带宽文件（5G Netflix）
            edge3_files = os.listdir(edge3_dir)
            bandwidth_files["edge3"] = os.path.join(edge3_dir, random.choice(edge3_files))
            rtt_values["edge3"] = 20  # 5G网络的典型RTT值
            
        except Exception as e:
            print(f"Error getting bandwidth files: {e}")
            # 使用默认带宽文件
            for server in ["edge1", "edge2", "edge3"]:
                bandwidth_files[server] = self.create_default_bandwidth_file()
                rtt_values[server] = -1
        
        return bandwidth_files, rtt_values

    def create_default_bandwidth_file(self):
        """创建默认带宽数据文件"""
        # 创建一个临时文件来存储默认带宽数据
        temp_dir = "../data/bandwidth/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, "default_bandwidth.txt")
        with open(file_path, 'w') as f:
            # 生成100个时间点的带宽数据
            for i in range(100):
                # 时间戳 带宽(bps)
                f.write(f"{i} 1000000\n")
        
        return file_path

    def step(self, action):
        """执行动作并获得奖励"""
        # 获取服务器索引和码率索引
        server_idx = action // 6  # 0,1,2 分别对应三个服务器
        bitrate_idx = action % 6  # 0-5 对应码率选项
        
        # 判断是否命中缓存：
        # 1. bitrate_idx不为5（不是miss动作）
        # 2. 且该动作在mask中为1（该码率在缓存中）
        hitFlag = bitrate_idx != 5
        
        server_name = f"edge{server_idx + 1}"
        
        # 获取当前的忙碌度
        busy = self.busyList[self.segmentNum % len(self.busyList)]
        
        # 执行动作并直接使用 client.py 中的 reward
        reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, \
        reward, reqBI, done, segmentNum = self.client.run(bitrate_idx, busy, hitFlag, server_name)
        
        # 处理奖励
        processed_reward = self.process_reward(reward)
        
        # 更新状态时正确处理last_server_id
        last_server_idx = self.current_state[5] * 2  # 还原未归一化的值
        self.current_state = [
            reqBitrate / BITRATES[-1],
            lastBitrate / BITRATES[-1],
            (buffer/1000 - 30) / 10,
            (hThroughput - throughput_mean) / throughput_std,
            (mThroughput - throughput_mean) / throughput_std,
            server_idx / 2,  # 当前服务器ID
            last_server_idx / 3  # 上一个服务器ID（使用未归一化的值）
        ]
        
        self.segmentNum = segmentNum
        return processed_reward, done

    def init_state(self):
        """初始化状态"""
        return {
            'reqBitrate': 0,
            'lastBitrate': 0,
            'bufferSize': 0,
            'hThroughput': 0,
            'mThroughput': 0
        }

    def get_state(self):
        # 获取当前服务器ID
        if self.client.current_server is None:
            server_id = 0  # 默认使用edge1
        else:
            server_id = int(self.client.current_server.split('edge')[1]) - 1
        
        # 获取上一个服务器ID
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
        """更新策略网络"""
        # 确保所有输入都是 float32 类型
        states = torch.stack([s.float() for s in states]).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)  # 动作应该是长整型
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # 计算优势值
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()
            advantages = rewards - values
            # 标准化优势值
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # PPO更新
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
                
                # 计算策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算值函数损失
                value_loss = F.mse_loss(values.squeeze(), mb_rewards)
                
                # 计算熵奖励
                entropy = dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # 使用较小的梯度裁剪阈值
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # 更新学习率
        self.scheduler.step()
        
        return total_policy_loss / PPO_EPOCHS, total_value_loss / PPO_EPOCHS

    def process_bandwidth(self, hThroughput, server_name):
        """使用LSTM处理带宽数据"""
        processed_throughput = self.bandwidth_lstms[server_name].process_throughput(hThroughput)
        return (processed_throughput - throughput_mean) / throughput_std

    def get_action_mask(self):
        """为每个服务器生成动作掩码"""
        mask = [1] * A_DIM  # A_DIM = 18
        
        # 为每个服务器生成掩码
        for server_idx in range(3):  # 3个服务器
            base_idx = server_idx * 6  # 每个服务器的起始索引
            
            # 随机决定缓存多少个版本
            randmCachedBICount = random.randint(1, 5)
            BI = [0, 1, 2, 3, 4]  # 5个码率版本
            randomCachedBI = random.sample(BI, randmCachedBICount)
            
            # 设置对应版本的掩码
            for bIndex in range(5):
                action_idx = base_idx + bIndex
                if bIndex not in randomCachedBI:
                    mask[action_idx] = 0
            
            # 确保miss动作总是可用
            mask[base_idx + 5] = 1
        
        return mask

    def run(self):
        """训练主循环"""
        print("开始训练...")
        episode = 0
        best_reward = float('-inf')
        patience_counter = 0
        running_reward = 0
        last_server_id = -1  # 初始化last_server_id
        
        while episode < MAX_EP:
            # 初始化环境
            self.videoName = self.get_video()
            self.bandwidth_file, self.rtt = self.getBandwidthFile()
            self.busyList = self.get_busyTrace()
            self.segmentNum = 0
            
            # 为这个episode生成一次动作掩码，并保存
            self.current_mask = self.get_action_mask()
            
            # 初始化client
            reqBI = self.client.init(self.videoName, self.bandwidth_file, self.rtt, self.bwType)
            
            # 初始化状态
            self.current_state = [
                reqBI / BITRATES[-1],  # 归一化的请求码率
                -1 / BITRATES[-1],     # 归一化的上一个码率
                0,                     # 归一化的缓冲区大小
                0,                     # 归一化的历史吞吐量
                0,                     # 归一化的预测吞吐量
                0,                     # 归一化的当前server_id
                -1/3                   # 归一化的上一个server_id
            ]
            
            # 初始化轨迹存储
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            episode_reward = 0
            episode_steps = 0
            
            # 收集轨迹
            while True:
                # 获取当前状态
                state = self.get_state()
                
                # 选择动作
                # exploration_rate = max(0.1, 1.0 - episode / MAX_EP)  # 随训练进行逐渐减少探索
                # deterministic = np.random.random() > exploration_rate
                deterministic = True

                action, log_prob, value = self.policy.choose_action(self.current_mask, state, deterministic)

                # 执行动作并获得奖励
                # reward, done = self.step(action)
                
                # 打印当前动作的奖励
                # print(f"Action {action} reward: {reward:.4f}")

                # 处理动作
                server_idx = action // 6  # 确定选择哪个edge server
                bitrate_action = action % 6  # 确定码率或miss
                
                # 获取对应的server名称
                server_name = f"edge{server_idx + 1}"
                
                # 在执行动作前保存当前server_id
                last_server_id = server_idx
                
                # 执行动作
                busy = self.busyList[self.segmentNum % len(self.busyList)]
                hitFlag = True if bitrate_action != 5 else False
                
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = \
                    self.client.run(bitrate_action, busy, hitFlag, server_name)
                reward = reward / 5
                #打印详细的训练状态
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
                
                # 使用LSTM处理带宽
                processed_throughput = self.process_bandwidth(hThroughput, server_name)
                
                # 更新状态
                self.current_state = [
                    reqBitrate / BITRATES[-1],
                    lastBitrate / BITRATES[-1],
                    (buffer/1000 - 30) / 10,
                    # processed_throughput,  # 使用LSTM处理后的带宽
                    (hThroughput - throughput_mean) / throughput_std,
                    (mThroughput - throughput_mean) / throughput_std,
                    server_idx / 2,  # 当前服务器ID
                    (last_server_id + 1) / 3  # 上一个服务器ID
                ]
                
                # 存储轨迹
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                
                # 累计episode奖励
                episode_reward += reward
                episode_steps += 1
                
                # 更新状态
                self.segmentNum = segNum
                
                # 检查是否结束episode
                if self.segmentNum >= self.client.segmentCount:
                    last_server_id = -1
                    break
                
                # 检查是否达到批量大小
                if len(states) >= BATCH_SIZE:
                    break
            
            
            # 更新运行时平均奖励（使用较小的更新率以增加稳定性）
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            
            # 记录训练信息
            if episode % 10 == 0:
                # 使用running_reward来决定是否需要衰减学习率
                self.scheduler.step(running_reward)
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Episode {episode}, Running Reward: {running_reward:.2f}, LR: {lr:.6f}")
                
                # 保存最佳模型
                if running_reward > best_reward:
                    best_reward = running_reward
                    if not PRINTFLAG:
                        torch.save(self.policy.state_dict(), 
                                 f"{self.model_dir}/model/best_model.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # 定期保存模型
            if episode % 100 == 0 and not PRINTFLAG:
                torch.save(self.policy.state_dict(), 
                         f"{self.model_dir}/model/model_{episode}.pth")
            
            
            # 早停检查（仅在最小训练轮数后生效）
            if episode >= MIN_EPISODES:
                if patience_counter >= PATIENCE and running_reward > EARLY_STOP_REWARD:
                    print(f"Early stopping at episode {episode}")
                    break

            episode += 1
        
        print("训练完成!")
        
        # 保存最终模型
        if not PRINTFLAG:
            torch.save(self.policy.state_dict(), 
                     f"{self.model_dir}/model/final_model.pth")


def main():
    # 创建结果保存目录
    if not PRINTFLAG:
        time_local = time.localtime(int(time.time()))
        dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
        model_dir = f"../data/RL_model/{dt}"
        os.makedirs(f"{model_dir}/model", exist_ok=True)
        
        # 重定向标准输出和错误到日志文件
        log_file = f"{model_dir}/training.log"
        sys.stdout = open(log_file, 'w', buffering=1)
        sys.stderr = sys.stdout
    
    # 创建worker并训练
    worker = Worker(model_dir)
    worker.run()

    # 保存训练结果
    if not PRINTFLAG:
        plt.figure(figsize=(10, 6))
        # 绘制原始reward曲线
        plt.plot(worker.rewards_history, 
                alpha=0.3, 
                color='blue', 
                label='Episode Reward')
        
        # 绘制移动平均线
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
        
        # 设置图表属性
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig(f"{model_dir}/training_curve.pdf")
        plt.close()

if __name__ == "__main__":
    main()