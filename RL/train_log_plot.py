import matplotlib.pyplot as plt
import re
import numpy as np

def parse_log_file(log_path):
    episodes = []
    rewards = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 使用正则表达式匹配episode和reward
            match = re.match(r'Episode (\d+), Running Reward: ([\d.]+)', line.strip())
            if match:
                episode = int(match.group(1))
                reward = float(match.group(2))
                episodes.append(episode)
                rewards.append(reward)
    
    return np.array(episodes), np.array(rewards)

# 读取并解析日志文件
log_path = "/home/zlp/lfbm-ppo/data/RL_model/2024-11-17_16-25-56/training.log"
episodes, rewards = parse_log_file(log_path)

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制奖励曲线
plt.plot(episodes, rewards, 'b-', alpha=0.6, label='Running Reward')

# 添加平滑曲线（使用移动平均）
window_size = 50
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(episodes[window_size-1:], smoothed_rewards, 'r-', 
         label=f'Smoothed Reward (window={window_size})')

# 设置图形属性
plt.xlabel('Episode')
plt.ylabel('Running Reward')
plt.title('Training Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 添加水平参考线
plt.axhline(y=180, color='g', linestyle='--', alpha=0.5, label='Target Reward')

# 设置y轴范围
plt.ylim(0, 600)

# 保存图形
plt.savefig("/home/zlp/lfbm-ppo/data/RL_model/2024-11-17_16-25-56/training_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print("曲线图已保存为 training_curve.png")
