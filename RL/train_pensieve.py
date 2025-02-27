import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from client_multi_server import Client
import random

# 环境参数
BITRATES = [350, 600, 1000, 2000, 3000]  # kbps
S_INFO = 6  # 状态维度：上一个码率、缓冲区大小、吞吐量历史、下载时间历史、剩余块数、下一个块大小
S_LEN = 8   # 状态历史长度
A_DIM = 5   # 码率选择动作数
M_IN_K = 1000

# 训练参数
ACTOR_LR = 1e-4   # Actor网络学习率 α = 10^-4
CRITIC_LR = 1e-3  # Critic网络学习率 α' = 10^-3
BATCH_SIZE = 512
MAX_EPISODES = 10000
GAMMA = 0.99  # 折扣因子 γ = 0.99，表示当前动作受未来100步的影响

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        # Actor网络
        self.actor = nn.ModuleList([
            nn.Conv1d(S_INFO, 128, 4, padding=1, padding_mode='replicate'),
            nn.Conv1d(128, 128, 4, padding=1, padding_mode='replicate'),
            nn.Conv1d(128, 128, 4, padding=1, padding_mode='replicate'),
            nn.Linear(640, 128),  # 修正: 输入维度是 128 * 5 = 640
            nn.Linear(128, A_DIM)
        ])
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(S_INFO * S_LEN, 128),  # 输入维度是 6 * 8 = 48
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        # 调整输入形状 [batch, S_INFO, S_LEN]
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        elif len(state.shape) == 1:
            state = state.view(1, S_INFO, S_LEN)
            
        # Actor网络前向传播
        x = state  # [batch, S_INFO=6, S_LEN=8]
        x = F.relu(self.actor[0](x))  # [batch, 128, 5]
        x = F.relu(self.actor[1](x))  # [batch, 128, 5]
        x = F.relu(self.actor[2](x))  # [batch, 128, 5]
        x = x.view(x.size(0), -1)     # [batch, 128 * 5]
        x = F.relu(self.actor[3](x))  # [batch, 128]
        logits = self.actor[4](x)     # [batch, A_DIM]
        
        # Critic网络前向传播
        state_flat = state.view(state.size(0), -1)  # [batch, S_INFO * S_LEN]
        value = self.critic(state_flat)             # [batch, 1]
        
        return F.softmax(logits, dim=-1), value

    def _debug_shapes(self, x):
        """用于调试的函数，打印每一层的输出形状"""
        print(f"Input shape: {x.shape}")
        x = F.relu(self.actor[0](x))
        print(f"After conv1: {x.shape}")
        x = F.relu(self.actor[1](x))
        print(f"After conv2: {x.shape}")
        x = F.relu(self.actor[2](x))
        print(f"After conv3: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class PensieveAgent:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.network = ActorCritic().to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': ACTOR_LR},
            {'params': self.network.critic.parameters(), 'lr': CRITIC_LR}
        ])
        
        # 添加学习率衰减
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,  # 每1000个episode衰减一次
            gamma=0.9  # 每次衰减到原来的90%
        )
        
        # 初始化replay buffer
        self.replay_buffer = ReplayBuffer(10000)
        
        # 初始化RTT字典
        self.rttDict = {
            1: 20,  # FCC
            2: 20,  # HSDPA
        }
        
        # 初始化Client
        self.client = Client("train", model_dir)
        
        # 初始化状态相关的属性
        self.throughput_history = []
        self.download_history = []
        self.buffer_size = 0
        self.last_bitrate = 0
        self.hThroughput = 0
        self.mthroughput = 0
        self.busy = 0
        
        # 初始化吞吐量统计参数
        self.throughput_mean = 2297514.2311790097
        self.throughput_std = 4369117.906444455
        
        # 熵因子β，从1衰减到0.1
        self.entropy_weight = 1.0
        self.entropy_weight_min = 0.1
        self.entropy_weight_decay = (1.0 - 0.1) / 100000
        
        self.best_reward = float('-inf')
        self.best_episode = 0
        
    def _get_state(self):
        """构建Pensieve原始的状态表示"""
        state = np.zeros((S_INFO, S_LEN))
        
        # 上一个码率
        state[0, -1] = self.last_bitrate / float(np.max(BITRATES))
        
        # 缓冲区大小
        state[1, -1] = self.buffer_size / 1000.0  # ms -> s
        
        # 吞吐量历史
        curr_idx = S_LEN - len(self.throughput_history)
        for i, throughput in enumerate(self.throughput_history):
            state[2, curr_idx + i] = throughput / M_IN_K / M_IN_K  # bytes/ms -> Mbps
        
        # 下载时间历史
        curr_idx = S_LEN - len(self.download_history)
        for i, download in enumerate(self.download_history):
            state[3, curr_idx + i] = download
        
        # 剩余块数
        state[4, -1] = self.client.segmentCount - self.client.segmentNum
        
        # 下一个块的大小
        if self.client.segmentNum < len(self.client.videoSizeList):
            next_chunk_sizes = self.client.videoSizeList[self.client.segmentNum]
            for i in range(len(BITRATES)):
                state[5, i] = next_chunk_sizes[i]
        
        return state
        
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).to(self.device)
        action_probs, _ = self.network(state)
        
        # 获取概率分布
        probs = action_probs.detach().cpu().numpy()
        
        # 使用动态epsilon-greedy策略
        epsilon = max(0.1, min(0.3, self.entropy_weight))
        
        if random.random() < epsilon:
            action = np.random.choice(len(BITRATES))
        else:
            action = torch.argmax(action_probs).item()
        
        return action
        
    def update(self, state, action, reward, next_state, done):
        """更新模型"""
        # 转换为tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # 计算当前动作概率和状态值
        action_probs, value = self.network(state)
        _, next_value = self.network(next_state)
        
        # 计算优势函数并裁剪
        advantage = reward + GAMMA * next_value * (1 - done) - value
        advantage = advantage.clamp(-10, 10)
        
        # 计算actor损失，确保数值稳定性
        selected_probs = action_probs.gather(1, action.unsqueeze(1))
        # 添加epsilon防止log(0)
        epsilon = 1e-10
        action_log_prob = torch.log(selected_probs.clamp(min=epsilon, max=1.0))
        actor_loss = -action_log_prob * advantage.detach()
        
        # 计算熵损失
        entropy = -(action_probs * torch.log(action_probs + epsilon)).sum(1)
        entropy_loss = -self.entropy_weight * entropy
        
        # 计算critic损失并裁剪
        critic_loss = advantage.pow(2).clamp(0, 100)
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.mean().backward()
        # 使用梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # 更新熵权重
        self.entropy_weight = max(
            self.entropy_weight_min,
            self.entropy_weight - self.entropy_weight_decay
        )
        
        return loss.item()

    def calculate_reward(self, bitrate, last_bitrate, rebuffer_time):
        """自定义奖励函数，注重码率和码率平滑性"""
        # 码率奖励：归一化的码率值
        bitrate_reward = bitrate / float(np.max(BITRATES))  # 归一化到[0,1]
        
        # 码率变化惩罚：相对变化的绝对值
        if last_bitrate > 0:
            bitrate_change = abs(bitrate - last_bitrate) / float(np.max(BITRATES))
            smoothness_penalty = -1.3 * bitrate_change  # 增加平滑性的权重
        else:
            smoothness_penalty = 0
        
        # rebuffer惩罚：保持较大的权重以避免卡顿
        rebuffer_penalty = -1.1 * rebuffer_time / 1000.0  # ms -> s
        
        # 总奖励
        reward = bitrate_reward + smoothness_penalty + rebuffer_penalty
        
        return reward

    def train(self):
        """训练主循环"""
        for episode in range(MAX_EPISODES):
            # 初始化环境
            video_name = self._get_video()
            bandwidth_file, rtt = self._get_bandwidth_file()
            bandwidth_files = {"edge1": bandwidth_file}
            rtt_dict = {"edge1": rtt}
            
            # 初始化client
            self.client.init(video_name, bandwidth_files, rtt_dict, self.bwType)
            
            # 获取busy列表
            busy_folder = "../data/trace/busy/2"
            busy_files = os.listdir(busy_folder)
            busy_file_path = os.path.join(busy_folder, random.choice(busy_files))
            
            self.busyList = []
            with open(busy_file_path, 'r') as f:
                busy_lines = f.readlines()
                for line in busy_lines:
                    self.busyList.append(float(line.strip()))
            
            state = self._get_state()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                busy = self.busyList[self.client.segmentNum % len(self.busyList)]
                
                # 使用client.run获取状态信息
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, rebuffer_time, _, done, _ = self.client.run(
                    action, 
                    busy, 
                    hitFlag=True,
                    server_name="edge1"
                )
                
                # 使用新的奖励函数
                reward = self.calculate_reward(reqBitrate, lastBitrate, rebuffer_time) / 5
                
                # 更新历史记录
                self.throughput_history.append(hThroughput)
                if len(self.throughput_history) > 8:
                    self.throughput_history.pop(0)
                    
                self.buffer_size = buffer
                self.last_bitrate = lastBitrate
                
                next_state = self._get_state()
                
                # 更新网络
                loss = self.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # 计算平均奖励
            #if step_count > 0:
            #    episode_reward = episode_reward / step_count  # 使用平均奖励
            
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {step_count}")
            
            # 在每个episode结束后更新学习率
            self.scheduler.step()
            
            # 打印当前学习率
            if episode % 100 == 0:
                current_lr_actor = self.optimizer.param_groups[0]['lr']
                current_lr_critic = self.optimizer.param_groups[1]['lr']
                print(f"Current learning rates - Actor: {current_lr_actor:.6f}, Critic: {current_lr_critic:.6f}")
            
            # 保存最优模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_episode = episode
                self._save_model(episode, is_best=True)
            
            # 定期保存检查点
            if episode % 100 == 0:
                self._save_model(episode)
    
    def _get_video(self):
        """获取视频文件"""
        video_file = open("../file/videoList/video.txt")
        video_list = video_file.readlines()
        video_list = [(i.split(" ")[0], float(i.split(" ")[1])) for i in video_list]
        
        video_random = random.random()
        for video_name, prob in video_list:
            if video_random < prob:
                return video_name
        return video_list[-1][0]
    
    def _get_bandwidth_file(self):
        """获取带宽文件 - 只使用edge1的带宽文件"""
        # edge1到client的带宽（可变：FCC, HSDPA, mine）
        self.bwType = random.randint(1, 3)
        if self.bwType == 1:
            edge1_dir = "../data/bandwidth/train/FCC"
        elif self.bwType == 2:
            edge1_dir = "../data/bandwidth/train/HSDPA"
        else:
            edge1_dir = "../data/bandwidth/train/mine"
        
        try:
            # 处理edge1的带宽文件
            if self.bwType == 3:
                ip_dirs = os.listdir(edge1_dir)
                ip = random.choice(ip_dirs)
                # if ip not in self.rttDict:
                #     print("no this ip:", ip, "in the bandwidth directory")
                #     edge1_dir = "../data/bandwidth/train/HSDPA"
                #     bandwidth_file = os.path.join(edge1_dir, random.choice(os.listdir(edge1_dir)))
                #     rtt = 20  # 默认RTT值
                # else:
                rtt = 20  # 默认RTT值
                bw_files = os.listdir(os.path.join(edge1_dir, ip))
                bandwidth_file = os.path.join(edge1_dir, ip, random.choice(bw_files))
            else:
                bw_files = os.listdir(edge1_dir)
                bandwidth_file = os.path.join(edge1_dir, random.choice(bw_files))
                rtt = self.rttDict.get(self.bwType, 20)  # 如果没有对应的RTT值，使用默认值20
                
        except Exception as e:
            print(f"Error getting bandwidth files: {e}")
            # 使用默认带宽文件
            bandwidth_file = self._create_default_bandwidth_file()
            rtt = 20
        
        return bandwidth_file, rtt

    def _create_default_bandwidth_file(self):
        """创建默认带宽数据文件"""
        temp_dir = "../data/bandwidth/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, "default_bandwidth.txt")
        with open(file_path, 'w') as f:
            # 生成100个时间点的带宽数据
            for i in range(100):
                # 时间戳 带宽(bps)
                f.write(f"{i} 1000000\n")
        
        return file_path
    
    def _save_model(self, episode, is_best=False):
        """保存模型"""
        checkpoint = {
            'episode': episode,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),  # 保存学习率调度器状态
            'reward': self.best_reward,
        }
        
        if not is_best:
            torch.save(checkpoint, 
                      f"{self.model_dir}/model/pensieve_model_{episode}.pth")
        
        if is_best:
            print(f"Saving best model with reward {self.best_reward:.2f} at episode {episode}")
            torch.save(checkpoint, f"{self.model_dir}/model/pensieve_model_best.pth")
            torch.save(checkpoint, "../data/RL_model/pensieve_model.pth")

def main():
    # 创建保存目录
    time_local = time.localtime(int(time.time()))
    dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
    model_dir = f"../data/Pensieve_Model/{dt}"
    os.makedirs(f"{model_dir}/model", exist_ok=True)
    
    # 创建并训练agent
    agent = PensieveAgent(model_dir)
    agent.train()

if __name__ == "__main__":
    main()