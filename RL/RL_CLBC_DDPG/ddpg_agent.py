import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Actor, Critic, OUNoise
from utils import map_actions, calculate_reward
import random

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 超参数
        self.batch_size = 32
        self.memory_capacity = 10000
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.001   # 软更新参数
        
        # 初始化记忆库
        self.memory = []
        self.memory_counter = 0
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # 探索噪声
        self.noise = OUNoise(action_dim)
    
    def store_transition(self, state, last_bitrate, action, reward, next_state, next_last_bitrate):
        """存储转换数据到记忆库"""
        # 确保所有数据都是numpy数组
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(last_bitrate, torch.Tensor):
            last_bitrate = last_bitrate.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(next_last_bitrate, torch.Tensor):
            next_last_bitrate = next_last_bitrate.cpu().numpy()
        
        transition = (state, last_bitrate, action, reward, next_state, next_last_bitrate)
        
        # 如果记忆库满了，替换旧的数据
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        
        self.memory.append(transition)
        self.memory_counter += 1
    
    def select_action(self, state, last_bitrate, add_noise=True):
        """选择动作"""
        # 检查输入是否已经是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(last_bitrate, torch.Tensor):
            last_bitrate = torch.FloatTensor([last_bitrate]).to(self.device)
        
        # 确保维度正确
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if last_bitrate.dim() == 0:
            last_bitrate = last_bitrate.unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state, last_bitrate).cpu().data.numpy().flatten()
        self.actor.train()
        
        if add_noise:
            action = np.clip(action + self.noise.sample(), 0, 1)
        
        return action
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 采样batch
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        last_bitrate_batch = torch.FloatTensor([t[1] for t in batch]).to(self.device)
        action_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[3] for t in batch]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        next_last_bitrate_batch = torch.FloatTensor([t[5] for t in batch]).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch, next_last_bitrate_batch)
            target_q = self.target_critic(next_state_batch, next_last_bitrate_batch, next_actions)
            target_q = reward_batch + self.gamma * target_q.unsqueeze(1)  # [B,1]
        
        # 计算当前Q值
        current_q = self.critic(state_batch, last_bitrate_batch, action_batch).unsqueeze(1)  # [B,1]
        
        # 确保维度匹配
        print(f"Debug - Before squeeze:")
        print(f"current_q shape: {current_q.shape}")
        print(f"target_q shape: {target_q.shape}")
        
        # 计算critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新actor
        actor_loss = -self.critic(state_batch, last_bitrate_batch, 
                                 self.actor(state_batch, last_bitrate_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
    
    def _soft_update(self, target, source):
        """软更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )