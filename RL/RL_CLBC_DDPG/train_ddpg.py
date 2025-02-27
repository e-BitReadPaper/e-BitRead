import sys
import os
import random
import numpy as np

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # RL目录
root_dir = os.path.dirname(parent_dir)     # lfbm-ppo目录
sys.path.append(parent_dir)                # 添加RL目录到Python路径

# 修正所有文件路径
VIDEO_LIST_PATH = os.path.join(root_dir, "file", "video.txt")
VIDEO_MPD_DIR = os.path.join(root_dir, "file", "video_mpd")
BANDWIDTH_DIR = os.path.join(root_dir, "data", "bandwidth", "train")
MODEL_SAVE_DIR = os.path.join(root_dir, "data", "RL_model", "DDPG")

# 创建必要的目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"Debug - Paths:")
print(f"Root dir: {root_dir}")
print(f"Video list path: {VIDEO_LIST_PATH}")
print(f"Video MPD dir: {VIDEO_MPD_DIR}")
print(f"Model save dir: {MODEL_SAVE_DIR}")

import torch
import numpy as np
from client import Client
from ddpg_agent import DDPGAgent
from utils import map_actions, calculate_reward
import time
import platform

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 环境参数
BITRATES = [500, 850, 1200, 1850]  # DDPG使用的码率集合
STATE_DIM = 5  # 状态维度
ACTION_DIM = 3  # 动作维度 (码率,目标缓冲,延迟限制)
MAX_EPISODES = 3000 if platform.system() == "Linux" else 40

# 在导入部分之后添加
def monkey_patch_client():
    """修补Client类的parseMPDFile方法，使其使用绝对路径"""
    def new_parse_mpd_file(self, videoName):
        try:
            mpd_path = os.path.join(self.videoMPDDir, videoName.split()[0], "stream.mpd")
            print(f"Reading MPD file from: {mpd_path}")
            with open(mpd_path, 'r') as f:
                responseStr = f.read()
            # ... 其他代码保持不变 ...
            return responseStr
        except Exception as e:
            print(f"Error reading MPD file: {e}")
            raise
    
    # 替换原方法
    Client.parseMPDFile = new_parse_mpd_file

# 在main函数之前调用
monkey_patch_client()

# 定义常量
BITRATES = [500, 850, 1200, 1850]  # Kbps
M_IN_K = 1000.0  # 转换系数

class DDPGTrainer:
    def __init__(self):
        # 状态空间维度：[throughput_k, download_time_k, chunk_size_k, buffer_k, if_hit_k]
        self.state_dim = 5 * 5  # 5个历史数据，每个5维
        self.action_dim = 1  # 动作空间维度：比特率选择索引
        
        # 初始化客户端
        self.client = Client(type="train")
        
        # 设置client的路径 - 使用绝对路径
        self.client.videoMPDDir = VIDEO_MPD_DIR
        self.client.videoSizeDir = os.path.join(root_dir, "file", "videoSize")
        self.client.video_file_path = VIDEO_LIST_PATH
        
        # 确保client不使用相对路径
        self.client.root_dir = root_dir
        
        # 打印路径进行调试
        print(f"\nClient paths:")
        print(f"Video file path: {self.client.video_file_path}")
        print(f"Video MPD dir: {self.client.videoMPDDir}")
        print(f"Video size dir: {self.client.videoSizeDir}")
        
        # 初始化DDPG代理
        self.agent = DDPGAgent(self.state_dim, self.action_dim, device)
        
        # 训练相关参数
        self.max_episodes = 3000
        self.save_interval = 100  # 每100个episode保存一次模型
        self.eval_interval = 50   # 每50个episode评估一次
        
        # 记录训练历史
        self.rewards_history = []
        self.eval_rewards_history = []
        self.episode_lengths = []
        
        # 创建保存目录
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        # 日志文件
        self.log_file = open(os.path.join(MODEL_SAVE_DIR, 'training_log.txt'), 'w')
        
        # 加载视频列表和码率版本信息
        self.video_list, self.video_info = self._load_video_list()
        if not self.video_list:
            raise ValueError("没有找到可用的视频")
        
        # 码率版本映射（1-5分别对应不同的目标码率）
        self.bitrate_mapping = {
            1: 500,   # 码率版本1对应500Kbps
            2: 850,   # 码率版本2对应850Kbps
            3: 1200,  # 码率版本3对应1200Kbps
            4: 1850,  # 码率版本4对应1850Kbps
            5: 2500   # 码率版本5对应2500Kbps
        }
    
    def _load_video_list(self):
        """加载视频列表和码率版本信息"""
        try:
            print(f"\n尝试加载视频列表:")
            print(f"路径: {VIDEO_LIST_PATH}")
            print(f"文件是否存在: {os.path.exists(VIDEO_LIST_PATH)}")
            
            if not os.path.exists(VIDEO_LIST_PATH):
                raise FileNotFoundError(f"视频列表文件不存在: {VIDEO_LIST_PATH}")
            
            video_list = []
            video_info = {}
            
            with open(VIDEO_LIST_PATH, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        video_name, prob, bitrate_ver = parts
                        video_list.append(video_name)
                        video_info[video_name] = (float(prob), int(bitrate_ver))
            
            if not video_list:
                raise ValueError(f"视频列表为空: {VIDEO_LIST_PATH}")
            
            print(f"成功加载 {len(video_list)} 个视频")
            
            # 验证视频文件是否存在
            valid_videos = []
            for video in video_list:
                mpd_path = os.path.join(VIDEO_MPD_DIR, video, "stream.mpd")
                if os.path.isfile(mpd_path):
                    valid_videos.append(video)
                else:
                    print(f"警告: 找不到视频 {video} 的MPD文件: {mpd_path}")
            
            if not valid_videos:
                raise ValueError("没有找到有效的视频文件")
            
            print(f"找到 {len(valid_videos)} 个有效视频")
            
            # 只保留有效视频的信息
            valid_video_info = {video: video_info[video] for video in valid_videos}
            
            return valid_videos, valid_video_info
            
        except Exception as e:
            print(f"加载视频列表时出错: {str(e)}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"尝试访问的路径: {VIDEO_LIST_PATH}")
            raise
    
    def _get_state_vector(self, client):
        """从client获取状态向量并正确格式化"""
        # 获取5个特征的历史数据
        features = [
            client.dict['throughputList_k'],
            client.dict['downloadTime_k'],
            client.dict['chunkSize_k'],
            client.dict['buffer_k'],
            client.dict['ifHit_k']
        ]
        
        # 确保每个特征都有5个历史值
        for i in range(len(features)):
            if len(features[i]) < 5:
                features[i] = [0] * 5
        
        # 转换为numpy数组并归一化
        state = np.array(features, dtype=np.float32)
        
        # 简单归一化
        normalizers = [1e6, 1.0, 1e6, 1e3, 1.0]  # 针对不同特征的归一化因子
        for i in range(len(features)):
            state[i] = state[i] / normalizers[i]
        
        return state
    
    def train_episode(self, video_name, training=True):
        """训练或评估一个episode"""
        try:
            # 随机选择一个带宽日志文件
            bw_dir = os.path.join(root_dir, "data", "bandwidth", "train", "edge1", "FCC")
            log_files = [f for f in os.listdir(bw_dir) if f.endswith('.log')]
            selected_log = random.choice(log_files)
            bandwidth_files = {
                "edge1": os.path.join(bw_dir, selected_log)
            }
            print(f"使用带宽文件: {bandwidth_files['edge1']}")
            
            # 初始化视频环境
            rtt_dict = {"edge1": 20}
            self.client.init(
                videoName=video_name,
                bandwidthFiles=bandwidth_files,
                rtt=rtt_dict,
                bwType="mixed"
            )
            
            episode_reward = 0
            current_state = self._get_state_vector(self.client)
            last_bitrate = 0
            
            # 获取该视频的目标码率
            bitrate_ver = self.video_info[video_name][1]
            target_bitrate = self.bitrate_mapping[bitrate_ver]
            
            while True:
                # 选择动作
                action = self.agent.select_action(
                    state=current_state,
                    last_bitrate=last_bitrate,
                    add_noise=training
                )
                
                # 打印原始动作信息
                print(f"\nDebug - Action info:")
                print(f"Raw action from agent: {action}")
                print(f"Action shape: {action.shape}")
                
                # 映射码率 [-1,1] -> [0,4]
                bitrate_index = int((action[0] + 1) * 4 / 2)
                bitrate_index = np.clip(bitrate_index, 0, 4)
                selected_bitrate = self.bitrate_mapping[bitrate_index + 1]
                
                # 打印映射后的动作信息
                print(f"Mapped actions:")
                print(f"  Bitrate: {selected_bitrate}Kbps (index: {bitrate_index})")
                
                # 运行一步
                self.client.run(
                    action=bitrate_index,
                    busy=0,
                    hitFlag=1,
                    server_name="edge1"
                )
                
                # 获取状态信息用于计算奖励
                rebuffer_time = self.client.rebufferTimeOneSeg  # 当前片段的重缓冲时间
                latency = self.client.downloadTime             # 使用下载时间作为延迟
                
                # 获取当前码率和上一个码率
                current_bitrate = self.bitrate_mapping[bitrate_index + 1]
                
                # 使用utils中的calculate_reward函数计算奖励
                reward = calculate_reward(
                    dbit_t=current_bitrate,      # 当前选择的码率
                    dbit_prev=last_bitrate,      # 上一个码率
                    rebuffer=rebuffer_time,      # 重缓冲时间
                    latency=latency              # 延迟
                )
                
                # 打印详细的调试信息
                print(f"\nState info:")
                print(f"  Buffer size: {self.client.bufferSize/1000:.2f}s")
                print(f"  Rebuffer time: {rebuffer_time:.3f}s")
                print(f"  Total rebuffer time: {self.client.totalRebufferTime/1000:.3f}s")
                print(f"  Download time: {latency:.3f}s")
                print(f"  Current bitrate: {current_bitrate}Kbps")
                print(f"  Previous bitrate: {last_bitrate}Kbps")
                print(f"  Reward: {reward:.3f}")
                
                next_state = self._get_state_vector(self.client)
                
                # 存储经验
                if training:
                    self.agent.store_transition(
                        current_state,
                        last_bitrate,
                        action,
                        reward,
                        next_state,
                        action
                    )
                    
                    if len(self.agent.memory) >= self.agent.batch_size:
                        self.agent.train()
                
                # 更新状态和上一个码率
                current_state = next_state
                last_bitrate = selected_bitrate  # 注意：这里存储实际码率值，而不是动作值
                episode_reward += reward
                
                # 训练
                if len(self.agent.memory) >= self.agent.batch_size:
                    self.agent.train()
                
                # 检查是否结束
                if self.client.segmentNum >= self.client.segmentCount:
                    break
            
            return episode_reward
            
        except Exception as e:
            print(f"训练episode时发生错误: {e}")
            raise
    
    def train(self):
        """训练DDPG模型"""
        print("开始训练DDPG模型...")
        try:
            for episode in range(self.max_episodes):
                # 根据概率权重选择视频
                weights = [self.video_info[video][0] for video in self.video_list]
                video_name = random.choices(self.video_list, weights=weights, k=1)[0]
                bitrate_ver = self.video_info[video_name][1]
                target_bitrate = self.bitrate_mapping[bitrate_ver]
                
                print(f"\nEpisode {episode+1}/{self.max_episodes}")
                print(f"选择视频: {video_name}")
                print(f"码率版本: {bitrate_ver} (目标码率: {target_bitrate}Kbps)")
                
                # 训练一个episode
                episode_reward = self.train_episode(video_name)
                self.rewards_history.append(episode_reward)
                
                # 打印训练信息
                print(f"Episode {episode}/{self.max_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Average Reward: {np.mean(self.rewards_history[-100:]):.2f}")
                
                # 记录到日志文件
                self.log_file.write(f"{episode}\t{episode_reward:.2f}\t"
                                  f"{np.mean(self.rewards_history[-100:]):.2f}\n")
                self.log_file.flush()
                
                # 定期保存模型
                if (episode + 1) % self.save_interval == 0:
                    self.save_model(f"model_episode_{episode+1}")
                
                # 定期评估
                if (episode + 1) % self.eval_interval == 0:
                    eval_reward = self.evaluate()
                    self.eval_rewards_history.append(eval_reward)
                    print(f"Evaluation at episode {episode+1}: {eval_reward:.2f}")
        
        except KeyboardInterrupt:
            print("\n训练被手动中断")
        except Exception as e:
            print(f"训练过程出错: {e}")
            raise
        finally:
            self.log_file.close()
            self.save_model("final_model")
            self.plot_training_history()
    
    def save_model(self, name):
        """保存模型"""
        path = os.path.join(MODEL_SAVE_DIR, f"{name}.pth")
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'rewards_history': self.rewards_history,
            'eval_rewards_history': self.eval_rewards_history
        }, path)
        print(f"模型已保存到: {path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
            
            # 绘制训练奖励
            plt.figure(figsize=(10, 5))
            plt.plot(self.rewards_history, label='Training Rewards')
            plt.plot(np.arange(0, len(self.rewards_history), self.eval_interval),
                    self.eval_rewards_history, label='Evaluation Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training History')
            plt.legend()
            plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history.png'))
            plt.close()
            
        except Exception as e:
            print(f"绘制训练历史时出错: {e}")
    
    def evaluate(self, num_episodes=5):
        """评估模型性能"""
        total_reward = 0
        for _ in range(num_episodes):
            video_name = random.choice(self.video_list)
            episode_reward = self.train_episode(video_name, training=False)
            total_reward += episode_reward
        return total_reward / num_episodes

def main():
    trainer = DDPGTrainer()
    trainer.train()

if __name__ == "__main__":
    main()