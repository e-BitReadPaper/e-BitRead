import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 状态序列的卷积层
        self.conv1 = nn.Conv1d(5, 64, kernel_size=3, padding=1)  # 输入通道5，输出通道64
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # 计算卷积后的特征维度
        self.conv_out_size = 128 * 5  # 128通道 * 5时间步
        
        # 全连接层
        self.fc1 = nn.Linear(self.conv_out_size + 1, 128)  # +1是last_bitrate
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, state_seq, last_bitrate):
        # 确保输入维度正确
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(state_seq)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 连接last_bitrate
        x = torch.cat([x, last_bitrate.view(-1, 1)], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出范围[-1,1]
        
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 状态序列的卷积层
        self.conv1 = nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # 计算卷积后的特征维度
        self.conv_out_size = 128 * 5
        
        # 全连接层
        self.fc1 = nn.Linear(self.conv_out_size + 1 + action_dim, 128)  # +1是last_bitrate
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, state_seq, last_bitrate, action):
        # 确保输入维度正确
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(state_seq)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 连接last_bitrate和action
        x = torch.cat([x, last_bitrate.view(-1, 1), action.view(-1, 1)], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class OUNoise:
    """Ornstein-Uhlenbeck噪声"""
    def __init__(self, action_dim, theta=0.15, sigma=0.2, mu=0.0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state