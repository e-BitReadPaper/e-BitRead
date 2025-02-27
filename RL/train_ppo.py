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
from client import Client
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
CLIP_EPSILON = 0.05         # PPO裁剪参数
MAX_GRAD_NORM = 0.5        # 梯度裁剪阈值

# 训练参数
BATCH_SIZE = 4096          # 减小批量，提高更新频率
MINI_BATCH_SIZE = 512      # 相应减小
PPO_EPOCHS = 4             # 增加每批数据的更新次数
LR = 1e-4                  # 降低学习率，提高稳定性
NUM_WORKERS = 4            # worker数量

# 早停参数
EARLY_STOP_REWARD = 400    # 提高目标奖励阈值
PATIENCE = 100             # 增加耐心值





def takeSecond(elem):
    return elem[1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # actor网络
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.bn1_a = nn.BatchNorm1d(200)  # 添加BN层
        self.linear_2_a = nn.Linear(200, 100)
        self.bn2_a = nn.BatchNorm1d(100)  # 添加BN层
        self.output_a = nn.Linear(100, A_DIM)
        
        # critic网络
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.bn1_c = nn.BatchNorm1d(200)  # 添加BN层
        self.linear_2_c = nn.Linear(200, 100)
        self.bn2_c = nn.BatchNorm1d(100)  # 添加BN层
        self.output_c = nn.Linear(100, 1)
        
        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                 self.linear_1_c, self.linear_2_c, self.output_c])
        self.distribution = torch.distributions.Categorical
        
        # 将模型移到GPU
        self.to(device)

    def forward(self, x):
        # 确保输入在GPU上
        x = x.to(device)
        
        # actor前向传播
        linear_1_a = self.linear_1_a(x)
        if x.size(0) > 1:  # 只在batch size > 1时使用BN
            linear_1_a = self.bn1_a(linear_1_a)
        linear_1_a = F.relu6(linear_1_a)
        
        linear_2_a = self.linear_2_a(linear_1_a)
        if x.size(0) > 1:  # 只在batch size > 1时使用BN
            linear_2_a = self.bn2_a(linear_2_a)
        linear_2_a = F.relu6(linear_2_a)
        
        logits = self.output_a(linear_2_a)
        
        # critic前向传播
        linear_1_c = self.linear_1_c(x)
        if x.size(0) > 1:  # 只在batch size > 1时使用BN
            linear_1_c = self.bn1_c(linear_1_c)
        linear_1_c = F.relu6(linear_1_c)
        
        linear_2_c = self.linear_2_c(linear_1_c)
        if x.size(0) > 1:  # 只在batch size > 1时使用BN
            linear_2_c = self.bn2_c(linear_2_c)
        linear_2_c = F.relu6(linear_2_c)
        
        values = self.output_c(linear_2_c)
        
        return logits, values

    def choose_action(self, mask, state):
        self.eval()
        # 将输入移到GPU
        state = torch.FloatTensor(state).to(device)
        mask = torch.tensor(mask, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            logits, value = self.forward(state)
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            masked_logits = masked_logits - masked_logits.max()
            probs = F.softmax(masked_logits, dim=-1)
            
            if torch.isnan(probs).any():
                print("Warning: NaN values in probabilities:", probs)
            
            dist = self.distribution(probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
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


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        
    def store(self, state, action, log_prob, value, reward, mask):
        # 存储transition
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        
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
        for start in range(0, len(self.states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # 先转换为numpy数组，再转换为tensor
            states = np.array([self.states[i] for i in batch_indices])
            actions = np.array([self.actions[i] for i in batch_indices])
            log_probs = np.array([self.log_probs[i] for i in batch_indices])
            returns = np.array([self.returns[i] for i in batch_indices])
            advantages = np.array([self.advantages[i] for i in batch_indices])
            
            yield {
                'states': torch.FloatTensor(states),
                'actions': torch.LongTensor(actions),
                'log_probs': torch.FloatTensor(log_probs),
                'returns': torch.FloatTensor(returns).view(-1),  # 确保维度正确
                'advantages': torch.FloatTensor(advantages).view(-1),  # 确保维度正确
                'masks': [self.masks[i] for i in batch_indices]
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
        
        # 环境相关初始化
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
        
    def process_reward(self, raw_reward):
        """处理奖励值"""
        # 调整奖励缩放因子
        REWARD_SCALE = 50.0  # 可以调整这个值
        
        # 归一化奖励
        reward = raw_reward / REWARD_SCALE
        
        # 调整裁剪范围
        reward = np.clip(reward, -5.0, 5.0)  # 缩小裁剪范围
        
        return reward
    
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
        
        bandwidth_paths = {
            "edge1": "../data/bandwidth/train/edge1/FCC",
            "edge2": "../data/bandwidth/train/edge2/HSDPA",
            "edge3": "../data/bandwidth/train/edge3/mine"
        }
        
        for server_name in ["edge1", "edge2", "edge3"]:
            try:
                bw_dir = os.path.abspath(bandwidth_paths[server_name])
                
                if not os.path.exists(bw_dir):
                    bandwidth_files[server_name] = self.create_default_bandwidth_file()
                    continue
                
                if server_name in ["edge1", "edge2"]:
                    bw_files = glob.glob(os.path.join(bw_dir, "*.log"))
                else:
                    ip_dirs = os.listdir(bw_dir)
                    if not ip_dirs:
                        bandwidth_files[server_name] = self.create_default_bandwidth_file()
                        continue
                        
                    selected_ip = random.choice(ip_dirs)
                    ip_dir = os.path.join(bw_dir, selected_ip)
                    bw_files = glob.glob(os.path.join(ip_dir, "*.log"))
                
                if not bw_files:
                    bandwidth_files[server_name] = self.create_default_bandwidth_file()
                else:
                    bandwidth_files[server_name] = random.choice(bw_files)
                
                rtt_values[server_name] = self.rttDict.get(self.bwType, -1)
                
            except Exception as e:
                bandwidth_files[server_name] = self.create_default_bandwidth_file()
                rtt_values[server_name] = -1
        
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
        # 根据动作判断是否命中缓存
        hitFlag = (action != 5)
        
        # 获取当前的忙碌度
        busy = self.busyList[self.segmentNum % len(self.busyList)]
        
        # 执行动作并直接使用 client.py 中的 reward
        reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, \
        reward, reqBI, done, segmentNum = self.client.run(action, busy, hitFlag)
        
        # 处理奖励
        processed_reward = self.process_reward(reward)
        
        # 更新状态
        self.current_state = {
            'reqBitrate': reqBitrate / BITRATES[-1],
            'lastBitrate': lastBitrate / BITRATES[-1],
            'bufferSize': (buffer/1000 - 30) / 10,
            'hThroughput': (hThroughput - throughput_mean) / throughput_std,
            'mThroughput': (mThroughput - throughput_mean) / throughput_std
        }
        
        self.segmentNum = segmentNum
        return processed_reward  # 直接返回 client.py 中计算的 reward

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
        """获取当前状态"""
        if self.client.current_server is None:
            server_id = 0  # 默认使用edge1
            last_server_id = -1
        else:
            try:
                server_id = int(self.client.current_server.split('edge')[1]) - 1
                last_server_id = -1 if self.client.last_server is None else \
                                int(self.client.last_server.split('edge')[1]) - 1
            except (AttributeError, IndexError, ValueError):
                print(f"Warning: Invalid server name format. Current: {self.client.current_server}, Last: {self.client.last_server}")
                server_id = 0
                last_server_id = -1
        
        state = {
            'reqBitrate': self.current_state[0],
            'lastBitrate': self.current_state[1],
            'bufferSize': self.current_state[2],
            'hThroughput': self.current_state[3],
            'mThroughput': self.current_state[4],
            'server_id': server_id / 2,  # 归一化
            'last_server_id': (last_server_id + 1) / 3  # 归一化
        }
        
        return np.array([
            state['reqBitrate'],
            state['lastBitrate'],
            state['bufferSize'],
            state['hThroughput'],
            state['mThroughput'],
            state['server_id'],
            state['last_server_id']
        ])

    def select_action(self, state):
        """选择动作并返回相关信息"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            dist = self.policy.distribution(probs)
            
            # 选择动作
            action = dist.sample()
            
            # 计算动作的对数概率
            action_log_prob = dist.log_prob(action)
            
            return (
                action.item(),  # 转换为Python数值
                action_log_prob.item(),  # 转换为Python数值
                value.item()  # 转换为Python数值
            )

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
            
        # PPO更新
        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                logits, values = self.policy(mb_states)
                new_probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(mb_actions)
                
                ratio = torch.exp(new_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), rewards[mb_indices])
                entropy = dist.entropy().mean()
                
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

    def profile_performance(self):
        """性能分析"""
        print("\n=== 性能分析开始 ===")
        
        # 初始化环境
        self.videoName = self.get_video()
        self.bandwidth_file, self.rtt = self.getBandwidthFile()
        self.busyList = self.get_busyTrace()
        self.segmentNum = 0
        reqBI = self.client.init(self.videoName, self.bandwidth_file, self.rtt, self.bwType)
        self.current_state = self.init_state()
        
        # 记录开始时间
        start_time = time.time()
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # 运行一个小批次的训练
            states = []
            actions = []
            rewards = []
            
            for _ in range(min(100, BATCH_SIZE)):
                state = self.get_state()
                action = self.select_action(state)
                reward = self.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
            # 更新一次策略
            if states:
                self.update_policy(states, actions, rewards)
                
        except Exception as e:
            print(f"性能分析时出错: {e}")
            raise
            
        finally:
            profiler.disable()
            duration = time.time() - start_time
            
            print(f"\n性能统计:")
            print(f"总用时: {duration:.2f}秒")
            print(f"样本理速度: {min(100, BATCH_SIZE)/duration:.1f} samples/s")
            print(f"GPU显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"GPU利用率: {torch.cuda.utilization()}%\n")
            
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)
            
        print("=== 性能分析完成 ===\n")

    def process_bandwidth(self, hThroughput, server_name):
        """使用LSTM处理带宽数据"""
        processed_throughput = self.bandwidth_lstms[server_name].process_throughput(hThroughput)
        return (processed_throughput - throughput_mean) / throughput_std

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
            
            # 收集轨迹
            while True:
                # 获取当前状态
                state = self.get_state()
                
                # 选择动作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    logits, value = self.policy(state_tensor)
                    probs = F.softmax(logits, dim=-1)
                    dist = self.policy.distribution(probs)
                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)
                
                # 转换为Python数值
                action_value = action.item()
                log_prob_value = action_log_prob.item()
                value_value = value.squeeze().item()
                
                # 处理动作
                server_idx = action_value // 6  # 确定选择哪个edge server
                bitrate_action = action_value % 6  # 确定码率或miss
                
                # 获取对应的server名称
                server_name = f"edge{server_idx + 1}"
                
                # 在执行动作前保存当前server_id
                last_server_id = server_idx
                
                # 执行动作
                busy = self.busyList[self.segmentNum % len(self.busyList)]
                hitFlag = True if bitrate_action != 5 else False
                
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = \
                    self.client.run(bitrate_action, busy, hitFlag, server_name)
                
                # 添加详细的训练状态打印
                print(f"Episode {episode:5d} | "
                      f"Step {len(states):4d} | "
                      f"Seg {segNum:4d} | "
                      f"ReqBR: {reqBitrate:5d} | "
                      f"LastBR: {lastBitrate:5d} | "
                      f"Buffer: {buffer/1000:6.2f}s | "
                      f"HTput: {hThroughput/1000000:6.2f}Mbps | "
                      f"MTput: {mThroughput/1000000:6.2f}Mbps | "
                      f"Reward: {reward:6.2f} | "
                      f"ReqBI: {reqBI} | "
                      f"Done: {done} | "
                      f"Server: {server_name}")
                
                # 使用LSTM处理带宽
                processed_throughput = self.process_bandwidth(hThroughput, server_name)
                
                # 更新状态
                self.current_state = [
                    reqBitrate / BITRATES[-1],
                    lastBitrate / BITRATES[-1],
                    (buffer/1000 - 30) / 10,
                    processed_throughput,  # 使用LSTM处理后的带宽
                    (mThroughput - throughput_mean) / throughput_std,
                    server_idx / 2,  # 当前服务器ID
                    (last_server_id + 1) / 3  # 上一个服务器ID
                ]
                
                # 存储轨迹
                states.append(state)
                actions.append(action_value)
                rewards.append(reward)
                log_probs.append(log_prob_value)
                values.append(value_value)
                
                # 累计episode奖励
                episode_reward += reward
                
                # 更新状态
                self.segmentNum = segNum
                self.current_state = [
                    reqBitrate / BITRATES[-1],
                    lastBitrate / BITRATES[-1],
                    (buffer/1000 - 30) / 10,
                    processed_throughput,  # 使用LSTM处理后的带宽
                    (mThroughput - throughput_mean) / throughput_std,
                    server_idx / 2,
                    (last_server_id + 1) / 3
                ]
                
                # 检查是否结束episode
                if self.segmentNum >= self.client.segmentCount:
                    last_server_id = -1
                    break
                
                # 检查是否达到批量大小
                if len(states) >= BATCH_SIZE:
                    break
            
            # 更新运行时平均奖励
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            
            # 记录和打印训练信息
            if episode % 10 == 0:
                print(f"Episode {episode}, Running Reward: {running_reward:.2f}")
                
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
            
            # 早停检查
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
        plt.plot(worker.rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.savefig(f"{model_dir}/training_curve.pdf")

if __name__ == "__main__":
    main()