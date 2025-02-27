# -*- coding: utf-8 -*

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from multi_client import Client
import random
import os
import numpy as np
import time
import sys #用于接收参数
import platform


M_IN_K = 1000
BITRATES = [350, 600, 1000, 2000, 3000]
if platform.system() == "Linux":
    PRINTFLAG = False
else:
    PRINTFLAG = True
S_LEN = 7
A_DIM = 18
PUREFLAF = False
throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SERVERS = ["edge1", "edge2", "edge3"]


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
        
    #     # 确保模型和所有输入都在同一个设备上
    #     self = self.to(device)
        
    #     # 将输入移到正确的设备上
    #     if isinstance(state, np.ndarray):
    #         state = torch.FloatTensor(state)
    #     state = state.to(device)
        
    #     if isinstance(mask, list):
    #         mask = torch.tensor(mask)
    #     mask = mask.to(device).bool()
        
    #     # 添加批次维度（如果需要）
    #     if len(state.shape) == 1:
    #         state = state.unsqueeze(0)
    #     if len(mask.shape) == 1:
    #         mask = mask.unsqueeze(0)
        
    #     with torch.no_grad():
    #         # 正确解包forward的返回值
    #         logits, value = self.forward(state)
            
    #         # 现在对logits进行操作
    #         masked_logits = logits.clone()
    #         masked_logits[~mask] = float('-inf')
    #         masked_logits = masked_logits - masked_logits.max()
    #         probs = F.softmax(masked_logits, dim=-1)
            
    #         #调试信息
    #         # print("\nAction probabilities after masking:")
    #         # for i, prob in enumerate(probs[0]):
    #         #     if prob > 0:
    #         #         server_idx = i // 6
    #         #         action_idx = i % 6
    #         #         server_name = ["edge1", "edge2", "edge3"][server_idx]
    #         #         print(f"Action {i} ({server_name}, {'miss' if action_idx == 5 else f'bitrate_{action_idx}'}): {prob:.4f}")
            
    #         if torch.isnan(probs).any():
    #             print("Warning: NaN values in probabilities:", probs)
            
    #         dist = self.distribution(probs[0])
    #         action = dist.sample()
    #         action_log_prob = dist.log_prob(action)
            
    #     return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()
    def choose_action(self, mask, state):
        """
        在测试阶段选择动作，不需要探索
        Args:
            mask: 动作掩码
            state: 当前状态
        Returns:
            action_idx: 选择的动作索引
            action_log_prob: 动作的对数概率
            value: 状态值
        """
        self.eval()  # 设置为评估模式
        
        # 确保模型和所有输入都在同一个设备上
        self = self.to(device)
        
        # 将输入移到正确的设备上并处理格式
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        state = state.to(device)
        
        if isinstance(mask, list):
            mask = torch.tensor(mask)
        mask = mask.to(device).bool()
        
        # 添加批次维度（如果需要）
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        
        with torch.no_grad():  # 测试时不需要梯度
            # 获取策略网络输出
            logits, value = self.forward(state)
            
            # 应用动作掩码
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            
            # 直接使用softmax获取概率分布
            probs = F.softmax(masked_logits, dim=-1)
            
            # 在测试时直接选择概率最高的动作
            action = torch.argmax(probs, dim=-1)
            
            # 计算动作的对数概率（用于记录）
            dist = self.distribution(probs)
            action_log_prob = dist.log_prob(action)
        
        return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class Worker(mp.Process):
    def __init__(self, model_dir, model_name, policy, trace_index, cache_files, bw_type, num_clients=10):
        super(Worker, self).__init__()
        
        # 添加servers定义
        self.servers = ["edge1", "edge2", "edge3"]
        
        # 保存基本参数
        self.model_dir = model_dir
        self.model_name = model_name
        self.policy = policy
        self.trace_index = trace_index
        self.cache_files = cache_files
        self.bw_type = bw_type
        self.num_clients = num_clients
         # 添加logitsDict初始化
        self.logitsDict = {}
        # 初始化神经网络
        self.lnet = Net()
        if policy == "RL":
            self.lnet.load_state_dict(torch.load(model_dir + model_name, map_location=device, weights_only=True))
        
        # 初始化缓存列表
        self.cached_lists = {server: {} for server in SERVERS}
        
        # 从缓存文件加载数据
        for server_name, cache_file in cache_files.items():
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            video_name = parts[0].strip()
                            version = int(parts[1].strip())
                            if video_name not in self.cached_lists[server_name]:
                                self.cached_lists[server_name][video_name] = set()
                            self.cached_lists[server_name][video_name].add(version)
        
        # 初始化RTT信息
        self.rttDict = {}
        get_rttDict(self.rttDict)

        # 初始化 rtt
        self.rtt = self.rttDict
        # 初始化busy trace
        self.busyTraceL = os.listdir("../../data/trace/busy/2")
        
        # 设置trace目录
        self.trace_dir = f"{model_dir}/test_trace_{trace_index}/{bw_type}/{policy}"
        os.makedirs(self.trace_dir, exist_ok=True)
        
        # 初始化客户端
        self.clients = []
        for i in range(num_clients):
            client = Client(
                type="test",
                traceDir=f"{self.trace_dir}/client_{i}",
                client_id=i
            )
            self.clients.append(client)
        
        # 初始化客户端统计信息
        self.client_stats = {i: {
            "total_reward": 0,
            "total_rebuffer": 0,
            "avg_bitrate": 0,
            "bitrate_changes": 0,
            "segments_played": 0,
            "cache_hits": 0,
            "total_segments": 0
        } for i in range(num_clients)}
        
        # 加载视频列表
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]
        video_file.close()
        
        # 加载视频轨迹
        videoTraceFile = open("../file/videoTrace.txt")
        self.videoTraceIndex = 0
        self.videoTraceList = videoTraceFile.readlines()
        self.videoTraceList = [i.strip() for i in self.videoTraceList]
        videoTraceFile.close()
        
        # 为每个服务器分配不同类型的带宽文件
        self.bandwidth_files = {
            "edge1": [],  # FCC/HSDPA/mine
            "edge2": [],  # 5G Netflix
            "edge3": []   # 5G Netflix
        }
        
        # 加载edge1的带宽文件（根据bw_type）
        if self.bw_type == "FCC":
            edge1_dir = "../data/bandwidth/test/FCC"
            if os.path.exists(edge1_dir):
                self.bandwidth_files["edge1"] = sorted(
                    [os.path.join(edge1_dir, f) for f in os.listdir(edge1_dir) if f.endswith('.log')]
                )
        elif self.bw_type == "HSDPA":
            edge1_dir = "../data/bandwidth/test/HSDPA"
            if os.path.exists(edge1_dir):
                self.bandwidth_files["edge1"] = sorted(
                    [os.path.join(edge1_dir, f) for f in os.listdir(edge1_dir) if f.endswith('.log')]
                )
        else:  # mine
            edge1_dir = "../data/bandwidth/test/mine"
            if os.path.exists(edge1_dir):
                for ip_dir in os.listdir(edge1_dir):
                    ip_path = os.path.join(edge1_dir, ip_dir)
                    if os.path.isdir(ip_path):
                        self.bandwidth_files["edge1"].extend(
                            sorted([os.path.join(ip_path, f) for f in os.listdir(ip_path) if f.endswith('.log')])
                        )
        
        # 加载edge2和edge3的5G Netflix带宽文件
        netflix_dir = "../data/bandwidth/train/5G_Neflix_static"
        if os.path.exists(netflix_dir):
            netflix_files = sorted([os.path.join(netflix_dir, f) for f in os.listdir(netflix_dir) if f.endswith('.log')])
            # 平均分配Netflix文件给edge2和edge3
            mid = len(netflix_files) // 2
            self.bandwidth_files["edge2"] = netflix_files[:mid]
            self.bandwidth_files["edge3"] = netflix_files[mid:]
        
        # 修改cached_lists的结构，存储视频名称和其对应的码率版本
        self.cached_lists = {
            "edge1": {},  # {video_name: set(version_numbers)}
            "edge2": {},
            "edge3": {}
        }
        
        # 从缓存文件加载数据
        # 遍历每个服务器及其对应的缓存文件
        for server_name, cache_file in cache_files.items():
            # 检查缓存文件是否存在
            if os.path.exists(cache_file):
                # 打开并读取缓存文件
                with open(cache_file, 'r') as f:
                    # 逐行读取文件内容
                    for line in f:
                        # 将每行按空格分割成部分
                        parts = line.strip().split()
                        # 确保每行至少有3个部分(视频名称、版本号、大小)
                        if len(parts) >= 3:
                            video_name = parts[0].strip()  # 第一部分是视频名称
                            version = int(parts[1].strip())  # 第二部分是码率版本号
                            # 检查服务器是否在缓存列表中
                            if server_name in self.cached_lists:
                                # 如果这个视频还没有在该服务器的缓存中,创建一个新的集合
                                if video_name not in self.cached_lists[server_name]:
                                    self.cached_lists[server_name][video_name] = set()
                                # 将该版本号添加到视频的版本集合中
                                self.cached_lists[server_name][video_name].add(version)


    def get_action_mask(self, videoName):
        """获取动作掩码，考虑具体的码率版本"""
        pure_video_name = videoName.strip()
        
        # 初始化所有动作为不可用
        mask = [0] * A_DIM
        # print(f"\nGenerating mask for video: {pure_video_name}")
        
        for server_idx, server_name in enumerate(SERVERS):
            cache_dict = self.cached_lists[server_name]
            base_idx = server_idx * 6
            # print(f"\nChecking {server_name}:")
            
            # 设置miss动作为可用
            mask[base_idx + 5] = 1
            
            if pure_video_name in cache_dict:
                versions = cache_dict[pure_video_name]
                #print(f"{server_name} has versions: {sorted(list(versions))}")
                # 设置缓存中的版本为可用
                for version in versions:
                    action_idx = base_idx + (version - 1)
                    mask[action_idx] = 1
                    # print(f"Setting version {version} (action_idx={action_idx}) as available")
            # else:
            #     print(f"{server_name} does not have this video")
            
            #print(f"Current mask after {server_name}: {mask}")
        
        #print(f"Final mask: {mask}")
        #print(f"Available actions: {[i for i, m in enumerate(mask) if m == 1]}")
        return mask

    
    def choose_action_lower(self, mask, reqBI):
        """选择低于或等于请求码率的最高码率版本"""
        best_action = -1
        best_bitrate = -1
        
        # 将reqBI转换为整数
        reqBI = int(reqBI)  # 添加这行
        # 遍历所有服务器和码率
        for server_idx in range(3):  # 3个服务器
            base_idx = server_idx * 6
            # 检查每个码率版本（除了miss动作）
            for bitrate_idx in range(reqBI + 1):  # 只考虑小于等于reqBI的码率
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:  # 如果这个动作可用
                    if bitrate_idx > best_bitrate:
                        best_action = action_idx
                        best_bitrate = bitrate_idx
        
        # 如果没找到合适的码率，返回任意一个可用的miss动作
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_closest(self, mask, reqBI):
        """选择最接近请求码率的版本"""
        best_action = -1
        min_dist = float('inf')
        
        # 遍历所有服务器和码率
        for server_idx in range(3):
            base_idx = server_idx * 6
            # 检查每个码率版本（除了miss动作）
            for bitrate_idx in range(5):  # 5个码率版本
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:
                    dist = abs(bitrate_idx - reqBI)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = action_idx
        
        # 如果没找到合适的码率，返回任意一个可用的miss动作
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_highest(self, mask, reqBI):
        """选择最高可用码率版本"""
        best_action = -1
        best_bitrate = -1
        
        # 遍历所有服务器和码率
        for server_idx in range(3):
            base_idx = server_idx * 6
            # 检查每个码率版本（除了miss动作）
            for bitrate_idx in range(5):  # 5个码率版本
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:
                    if bitrate_idx > best_bitrate:
                        best_action = action_idx
                        best_bitrate = bitrate_idx
        
        # 如果没找到合适的码率，返回任意一个可用的miss动作
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_prefetch(self, mask, reqBI):
        """优先选择请求码率，如果不可用则随机选择miss或提前获取（带成本）"""
        # 将reqBI转换为整数
        reqBI = int(reqBI)
        # 首先尝试在每个服务器上找到请求的码率
        for server_idx in range(3):
            action_idx = server_idx * 6 + reqBI
            if mask[action_idx] == 1:
                return action_idx, False  # 返回动作和是否是prefetch的标记
        
        # 如果找不到请求的码率，随机决定是否使用miss
        if random.random() < 0.5:
            # 返回任意一个可用的miss动作
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx, False
        else:
            # 50%概率提前取到请求的码率，但需要增加下载时间
            return reqBI, True  # 返回动作和prefetch标记

    def choose_action_by_policy(self, mask, reqBI, policy):
        """根据不同策略选择动作
        Args:
            mask: 动作掩码
            reqBI: 请求的码率索引
            policy: 策略名称
        Returns:
            action: 选择的动作
        """
        if policy == "no_policy":
            # 返回第一个可用的miss动作
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
            
        elif policy == "lower":
            return self.choose_action_lower(mask, reqBI)
            
        elif policy == "closest":
            return self.choose_action_closest(mask, reqBI)
            
        elif policy == "highest":
            return self.choose_action_highest(mask, reqBI)
            
        elif policy == "prefetch":
            action, is_prefetch = self.choose_action_prefetch(mask, reqBI)
            return action
            
        else:
            raise ValueError(f"Unknown policy: {policy}")

    def getBandwidthFileList(self):
        rtt = -1
        if PUREFLAF:
            dir = "../../data/bandwidth/train/" + self.bwType
        else:

            dir = "../../data/bandwidth/test/" + self.bwType

        if self.bwType == "mine":
            ipDirList = os.listdir(dir)
            for ip in ipDirList:
                if ip not in self.rttDict:
                    print("no this ip:", ip, "in the bandwidth directory")
                    dir = "../../data/bandwidth/test/HSDPA"
                else:
                    rtt = self.rttDict[ip]
                bandwidthFileList = os.listdir(dir + "/" + ip)
                for fileName in bandwidthFileList:
                    self.bandwidth_fileList.append([dir + "/" + ip + "/" + fileName, rtt, self.bwType])
        else:
            bandwidthFileList = os.listdir(dir)
            for fileName in bandwidthFileList:
                self.bandwidth_fileList.append([dir + "/" + fileName, rtt, self.bwType])


    def get_busyTrace(self):
        """获取busy trace"""
        busyList = []
        busyTrace = random.choice(self.busyTraceL)
        busyTrace = "../../data/trace/busy/2/" + busyTrace
        lines = open(busyTrace).readlines()
        for line in lines:
            busyList.append(float(line.strip()))
        return busyList


    def getSizeDict():
        file = open("./file/videoSizePerVision.txt")
        lines = file.readlines()
        virsionSizeDict = {}  # videoName_bIndex size
        for line in lines:
            virsion = line.split(" ")[0]
            size = int(line.split(" ")[1])
            if virsion not in virsionSizeDict:
                virsionSizeDict[virsion] = size
            else:
                print(virsion, "show twice")
        return virsionSizeDict


    def saveLogits(self, videoName, logits):
        # 保存logits的方法
        key = videoName
        if key not in self.logitsDict:
            self.logitsDict[key] = []
        self.logitsDict[key].append(logits)
        
        # 可选：定期保存到文件
        if len(self.logitsDict[key]) % 100 == 0:  # 每100个logits保存一次
            save_dir = os.path.join(self.model_dir, "logits")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{key}_logits.npy")
            np.save(save_path, np.array(self.logitsDict[key]))


    def run(self):
        # 初始化每个服务器的奖励统计
        server_rewards = {server: {"sum": 0, "count": 0} for server in SERVERS}
        r_avg_sum = 0
        
        # 创建结果目录
        os.makedirs(self.trace_dir, exist_ok=True)
        resFile = open(f"{self.trace_dir}/test_result_{self.bw_type}_{self.policy}.txt", "w")
        
        for file_index in range(len(self.bandwidth_files['edge1'])):
            # 为每个客户端获取当前的带宽文件
            current_bandwidth_files = {
                "edge1": self.bandwidth_files["edge1"][file_index] if file_index < len(self.bandwidth_files["edge1"]) else self.bandwidth_files["edge1"][0],
                "edge2": self.bandwidth_files["edge2"][file_index % len(self.bandwidth_files["edge2"])],
                "edge3": self.bandwidth_files["edge3"][file_index % len(self.bandwidth_files["edge3"])]
            }
            
            # 为每个客户端获取不同的视频
            video_names = []
            for _ in range(self.num_clients):
                if self.videoTraceIndex >= len(self.videoTraceList):
                    self.videoTraceIndex = 0
                video_names.append(self.videoTraceList[self.videoTraceIndex].strip())
                self.videoTraceIndex += 1
            
            # 获取busy trace
            busyList = self.get_busyTrace()
            
            # 初始化每个客户端的状态
            client_states = []
            for i, client in enumerate(self.clients):
                reqBI = client.init(
                    videoName=video_names[i],
                    bandwidthFiles=current_bandwidth_files,  # 使用当前的带宽文件
                    rtt=self.rtt,
                    bwType=self.bw_type
                )
                # 初始化时设置 last_server_id 为 -1
                client.last_server_id = -1  # 添加这一行
                client_states.append({
                    'done': False,
                    'state': [
                        BITRATES[reqBI] / BITRATES[-1],
                        0,  # lastBitrate
                        0,  # buffer
                        throughput_mean,  # hThroughput
                        throughput_mean,  # mThroughput
                        0,  # server_idx
                        -1/3  # last_server_id
                    ],
                    'segNum': 1,  # 初始化片段编号
                    'r_sum': 0,
                    'total_step': 0
                })
            
            # 运行所有客户端直到所有视频都播放完成
            while not all(state['done'] for state in client_states):
                for i, (client, state) in enumerate(zip(self.clients, client_states)):
                    if not state['done']:
                        # 获取动作掩码
                        mask = self.get_action_mask(video_names[i])
                        
                        # 根据策略选择动作
                        if self.policy == "RL":
                            state_tensor = v_wrap(np.array(state['state'])[None, :])
                            a, logits, _ = self.lnet.choose_action(mask, state_tensor)
                            self.saveLogits(video_names[i], logits)
                        else:
                            # 使用其他策略选择动作
                            a = self.choose_action_by_policy(mask, state['state'][0], self.policy)
                        
                        # 执行动作
                        busy = busyList[state['segNum'] % len(busyList)]
                        hitFlag = a % 6 != 5
                        server_name = SERVERS[a // 6]
                        
                        reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, new_segNum = \
                            client.run(a % 6, busy, hitFlag, server_name)
                        
                        server_idx = a // 6
                        client.last_server_id = server_idx  # 添加这一行
                        # 更新状态
                        state['state'] = [
                            reqBitrate / BITRATES[-1],
                            lastBitrate / BITRATES[-1],
                            (buffer/1000 - 30) / 10,
                            (hThroughput - throughput_mean) / throughput_std,
                            (mThroughput - throughput_mean) / throughput_std,
                            (a // 6) / 2,
                            (client.last_server_id + 1) / 3
                        ]
                        state['done'] = done
                        state['segNum'] = new_segNum  # 更新片段编号
                        state['r_sum'] += reward
                        state['total_step'] += 1
                        
                        # 更新统计信息
                        self.client_stats[i]["total_reward"] += reward
                        self.client_stats[i]["total_rebuffer"] += client.rebufferTimeOneSeg
                        self.client_stats[i]["avg_bitrate"] += reqBitrate
                        self.client_stats[i]["total_segments"] += 1
                        if hitFlag:
                            self.client_stats[i]["cache_hits"] += 1
                        if self.client_stats[i]["segments_played"] > 0:
                            if abs(reqBitrate - lastBitrate) > 0:
                                self.client_stats[i]["bitrate_changes"] += 1
                        self.client_stats[i]["segments_played"] += 1
                        
                        # 更新服务器统计
                        server_rewards[server_name]["sum"] += reward
                        server_rewards[server_name]["count"] += 1
            
            # 输出进度信息
            print(f"\nProgress: {file_index + 1}/{len(self.bandwidth_files['edge1'])}")
            self.print_current_stats()
        
        # 保存最终统计结果
        self.save_final_stats()
        resFile.close()

    def print_current_stats(self):
        """打印当前统计信息"""
        print("\nCurrent Statistics:")
        for client_id, stats in self.client_stats.items():
            if stats["segments_played"] > 0:
                avg_reward = stats["total_reward"] / stats["segments_played"]
                avg_bitrate = stats["avg_bitrate"] / stats["segments_played"]
                cache_hit_rate = stats["cache_hits"] / stats["total_segments"] * 100
                
                print(f"\nClient {client_id}:")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Bitrate: {avg_bitrate:.2f} Kbps")
                print(f"  Cache Hit Rate: {cache_hit_rate:.2f}%")
                print(f"  Total Rebuffer Time: {stats['total_rebuffer']:.2f}s")
                print(f"  Segments Played: {stats['segments_played']}")

    def save_final_stats(self):
        """保存最终统计结果"""
        stats_file = f"{self.trace_dir}/final_stats_{self.bw_type}_{self.policy}.txt"
        with open(stats_file, 'w') as f:
            f.write("Client ID\tAvg Reward\tAvg Bitrate\tCache Hit Rate\tRebuffer Time\tSegments\n")
            for client_id, stats in self.client_stats.items():
                if stats["segments_played"] > 0:
                    avg_reward = stats["total_reward"] / stats["segments_played"]
                    avg_bitrate = stats["avg_bitrate"] / stats["segments_played"]
                    cache_hit_rate = stats["cache_hits"] / stats["total_segments"] * 100
                    f.write(f"{client_id}\t{avg_reward:.2f}\t{avg_bitrate:.2f}\t{cache_hit_rate:.2f}\t"
                           f"{stats['total_rebuffer']:.2f}\t{stats['segments_played']}\n")


def main():
    bwTypes = ["FCC", "HSDPA", "mine"]
    modelDir = "../data/RL_model/2024-12-19_14-30-50"
    modelName = "/model/best_model.pth"
    policys = ["RL", "no_policy", "lower", "closest", "highest", "prefetch"]
    policys = ["no_policy"]
    traceIndex = "multi_client_0"
    
    # 定义每个服务器的缓存文件
    cache_files = { 
        "edge1": "../file/cachedFile/cachedFile_20190612_pure.txt",
        "edge2": "../file/cachedFile/cachedFile_20241210.txt",
        "edge3": "../file/cachedFile/cachedFile_20241211.txt"
    }

    for bwType in bwTypes:
        for policy in policys:
            # 创建包含10个客户端的worker
            worker = Worker(
                model_dir=modelDir,
                model_name=modelName,
                policy=policy,
                trace_index=traceIndex,
                cache_files=cache_files,
                bw_type=bwType,
                num_clients=10
            )
            worker.run()


if __name__ == "__main__":
    main()