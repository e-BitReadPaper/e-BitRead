import numpy as np

def map_actions(actions):
    """将[-1,1]范围的动作映射到实际决策空间"""
    # 视频码率映射
    bitrates = np.array([500, 850, 1200, 1850, 2500])
    bit_idx = int((actions[0] + 1) * len(bitrates) / 2)
    bit_idx = np.clip(bit_idx, 0, len(bitrates)-1)
    dbit = bitrates[bit_idx]
    
    # 目标缓冲映射
    dtar = 0.5 if actions[1] < 0 else 1.0
    
    # 延迟限制映射 (假设范围是[0.5s, 2s])
    dlat = 0.5 + (actions[2] + 1) * 0.75  # 映射到[0.5, 2]
    
    return dbit, dtar, dlat

def calculate_reward(dbit_t, dbit_prev, rebuffer, latency):
    """计算奖励"""
    # 延迟惩罚系数
    alpha = 0.005 if latency < 1.0 else 0.01
    
    # 码率变化惩罚
    smoothness_penalty = 0.02 * abs(dbit_t - dbit_prev)
    
    # 总奖励
    reward = (dbit_t - 1.85 * rebuffer - alpha * latency - smoothness_penalty)
    
    return reward