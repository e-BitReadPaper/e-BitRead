# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('Agg')
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
        self.bn1_a = nn.BatchNorm1d(200)
        self.linear_2_a = nn.Linear(200, 100)
        self.bn2_a = nn.BatchNorm1d(100)
        self.output_a = nn.Linear(100, A_DIM)
        
        # critic网络
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.bn1_c = nn.BatchNorm1d(200)
        self.linear_2_c = nn.Linear(200, 100)
        self.bn2_c = nn.BatchNorm1d(100)
        self.output_c = nn.Linear(100, 1)
        
        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                 self.linear_1_c, self.linear_2_c, self.output_c])
        self.distribution = torch.distributions.Categorical
        
        self.to(device)

    def forward(self, x):
        linear_1_a = F.relu6(self.linear_1_a(x))
        linear_2_a = F.relu6(self.linear_2_a(linear_1_a))
        logits = self.output_a(linear_2_a)

        linear_1_c = F.relu6(self.linear_1_c(x))
        linear_2_c = F.relu6(self.linear_2_c(linear_1_c))
        values = self.output_c(linear_2_c)

        return logits, values


    def choose_action(self, mask, s):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).to(device)
            logits, _ = self.forward(state_tensor)
            
            # 处理动作掩码
            mask_tensor = torch.tensor(mask).to(device)
            masked_logits = logits.masked_fill(~mask_tensor.bool(), float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)
            
            # # 在选择动作前打印mask
            # print(f"Current mask: {mask}")

            # 选择最优动作
            action = probs.argmax().item()
            
            # 转换为edge server和具体动作
            server_idx = action // 6
            action_idx = action % 6
            
            return action_idx, logits.cpu().numpy()[0]


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class Worker(mp.Process):
    def __init__(self, model_dir, model_name, policy, trace_index, cache_files, bw_type):
        super(Worker, self).__init__()
        
        # 保存基本参数
        self.model_dir = model_dir
        self.model_name = model_name
        self.policy = policy
        self.trace_index = trace_index
        self.cache_files = cache_files
        self.bw_type = bw_type
        
        # 添加调试标志
        self.debug = False  # 默认关闭调试输出
        
        # 初始化神经网络
        self.lnet = Net()
        if policy == "RL":
            self.lnet.load_state_dict(torch.load(model_dir + model_name, map_location=device))
        
        # 初始化logitsDict
        self.logitsDict = {}
        
        # 初始化videoTraceIndex
        self.videoTraceIndex = 0
        
        # 获取RTT信息
        self.rttDict = {}
        get_rttDict(self.rttDict)
        
        # 初始化busy trace
        self.busyTraceL = os.listdir("../../data/trace/busy/2")
        
        # 设置trace目录
        trace_dir = f"{model_dir}/test_trace_{trace_index}/{bw_type}/{policy}"
        
        # 初始化客户端，设置type为"test"并传入trace_dir
        client_type = "mixed"  # 默认值
        if self.bw_type == "FCC":
            client_type = "FCC"
        elif self.bw_type == "HSDPA":
            client_type = "HSDPA"
        elif self.bw_type == "mine":
            client_type = "mine"
        
        # 修改这里：将type设置为"test"，并传入trace_dir
        self.client = Client(
            type="test",  # 改为"test"以启用trace记录
            traceDir=trace_dir  # 添加trace目录
        )
        
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
            "edge1": [],  # FCC
            "edge2": [],  # HSDPA
            "edge3": []   # mine
        }
        
        # 加载FCC带宽文件 (edge1)
        fcc_dir = "../data/bandwidth/train/edge1/FCC"
        if os.path.exists(fcc_dir):
            self.bandwidth_files["edge1"] = sorted(
                [os.path.join(fcc_dir, f) for f in os.listdir(fcc_dir) if f.endswith('.log')]
            )
        
        # 加载HSDPA带宽文件 (edge2)
        hsdpa_dir = "../data/bandwidth/train/edge2/HSDPA"
        if os.path.exists(hsdpa_dir):
            self.bandwidth_files["edge2"] = sorted(
                [os.path.join(hsdpa_dir, f) for f in os.listdir(hsdpa_dir) if f.endswith('.log')]
            )
        
        # 加载mine带宽文件 (edge3)
        mine_base_dir = "../data/bandwidth/train/edge3/mine"
        if os.path.exists(mine_base_dir):
            for ip_dir in os.listdir(mine_base_dir):
                ip_path = os.path.join(mine_base_dir, ip_dir)
                if os.path.isdir(ip_path):
                    self.bandwidth_files["edge3"].extend(
                        sorted([os.path.join(ip_path, f) for f in os.listdir(ip_path) if f.endswith('.log')])
                    )
        
        # 检查并打印带宽文件信息
        for edge, files in self.bandwidth_files.items():
            bw_type = "FCC" if edge == "edge1" else "HSDPA" if edge == "edge2" else "mine"
            # print(f"\nBandwidth files for {edge} ({bw_type}): {len(files)} files found")
            # if len(files) > 0:
            #     print(f"First file: {files[0]}")
            #     print(f"Last file: {files[-1]}")
        
        # 修改cached_lists的结构，存储视频名称和其对应的码率版本
        self.cached_lists = {
            "edge1": {},  # {video_name: set(version_numbers)}
            "edge2": {},
            "edge3": {}
        }
        
        # 从缓存文件加载数据
        for server_name, cache_file in cache_files.items():
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            video_name = parts[0]  # 只取视频名称
                            version = int(parts[1])  # 码率版本号
                            if server_name in self.cached_lists:
                                if video_name not in self.cached_lists[server_name]:
                                    self.cached_lists[server_name][video_name] = set()
                                self.cached_lists[server_name][video_name].add(version)
                            
                # # 打印调试信息
                # print(f"\n服务器 {server_name} 的缓存视频:")
                # for video, versions in self.cached_lists[server_name].items():
                #     print(f"{video}: 版本 {sorted(list(versions))}")
            # else:
            #     print(f"Warning: Cache file not found for {server_name}: {cache_file}")
        
        # # 打印缓存信息
        # for server, cached_videos in self.cached_lists.items():
        #     print(f"\nCached videos for {server}: {len(cached_videos)} videos")
        #     if len(cached_videos) > 0:
        #         print(f"Sample cached videos: {list(cached_videos)[:5]}")

    def load_cache_list(self, cache_file):
        """改进的缓存加载逻辑"""
        cached_dict = {}
        try:
            with open(cache_file, 'r') as f:
                for line in f:
                    if line.strip():
                        video_name, bitrate_version, size = line.strip().split()
                        key = video_name
                        if key not in cached_dict:
                            cached_dict[key] = set()
                        cached_dict[key].add(int(bitrate_version))  # 存储码率版本
        except Exception as e:
            print(f"Error loading cache file {cache_file}: {e}")
        return cached_dict

    def get_action_mask(self, videoName):
        """获取动作掩码，考虑具体的码率版本"""
        # 提取纯视频名称（去除概率值）
        pure_video_name = videoName.split()[0]
        
        mask = [0] * A_DIM
        # print(f"\n为视频 {videoName} (纯名称: {pure_video_name}) 生成动作掩码:")
        
        for server_idx, server_name in enumerate(SERVERS):
            cache_dict = self.cached_lists[server_name]
            base_idx = server_idx * 6
            
            # 使用纯视频名称检查缓存
            if pure_video_name in cache_dict:
                versions = cache_dict[pure_video_name]
                # print(f"服务器 {server_name} 缓存了视频 {pure_video_name} 的版本: {versions}")
                for version in versions:
                    if 1 <= version <= 5:  # 确保版本号在有效范围内
                        action_idx = base_idx + (version - 1)
                        mask[action_idx] = 1
                        # print(f"启用动作 {action_idx} (服务器:{server_name}, 版本:{version})")
            else:
                miss_idx = base_idx + 5
                mask[miss_idx] = 1
                # print(f"服务器 {server_name} 未缓存视频 {pure_video_name}, 启用miss动作 {miss_idx}")
        
        # print(f"最终掩码: {mask}")
        available_actions = [i for i, m in enumerate(mask) if m == 1]
        # print(f"可用动作列表: {available_actions}")
        return mask

    def choose_action_lower(self, choosableList, reqBI):
        for i in range(reqBI, -1, -1):
            if choosableList[i] == 1:
                return i
        return 5


    def choose_action_closest(self, choosableList, reqBI):
        dist = []
        for i in range(len(choosableList) - 1):
            if choosableList[i] == 1:
                dist.append([i, abs(reqBI - i)])
        if len(dist) == 0:
            return 5
        dist.sort(key=takeSecond)
        return dist[0][0]


    def choose_action_highest(self, choosableList, reqBI):
        for i in range(len(choosableList) - 2, -1, -1):
            if choosableList[i] == 1:
                return i
        return 5


    def choose_action_prefetch(self, mask, reqBI):
        if mask[reqBI] == 1:
            return reqBI
        else:
            rand = random.random()
            if rand < 0.5:
                return reqBI
            else:
                return 5



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


    def genRLCachedFile(self):
        resultDir = '../file/cachedFile'
        result_file_name = resultDir + '/' + self.cachedVideo.split('.')[0] + '_rl.txt'
        virsionSizeDict = self.getSizeDict()

        for key in self.logitsDict:
            logits = self.logitsDict[key]
            size = virsionSizeDict[key]
            costBenifit = logits / size * 1000000
            virsionList.append([key, costBenifit, logits, size])

        virsionList.sort(key=takeSecond, reverse=True)

        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y%m%d", time_local)

        file = open(result_file_name, 'w')
        cachedSize = 0
        i = 0
        cachedVirsionList = []
        while cachedSize < 10 * 1024 * 1024 * 1024:  # 10GB
            cachedSize += virsionList[i][3]
            cachedVirsionList.append(virsionList[i][0].split("_")[0] + " " + virsionList[i][0].split("_")[1])
            i += 1

        for virsion in cachedVirsionList:
            file.write(virsion + " " + str(chunkCountDict[virsion.split(" ")[0]]) + "\n")
            file.flush()
        file.close()


    def run(self):
        # 初始化每个服务器的奖励统计
        server_rewards = {
            "edge1": {"sum": 0, "count": 0},
            "edge2": {"sum": 0, "count": 0},
            "edge3": {"sum": 0, "count": 0}
        }
        
        r_avg_sum = 0
        resFile = open(f"{self.model_dir}/test_result_{self.bw_type}_{self.policy}.txt", "w")
        
        for file_index in range(len(self.bandwidth_files['edge1'])):
            # 获取busy trace
            busyList = self.get_busyTrace()
            
            # 构建每个服务器对应的带宽文件
            bandwidth_files = {
                "edge1": self.bandwidth_files["edge1"][file_index],
                "edge2": self.bandwidth_files["edge2"][file_index % len(self.bandwidth_files["edge2"])],
                "edge3": self.bandwidth_files["edge3"][file_index % len(self.bandwidth_files["edge3"])]
            }
            
            # 构建RTT字典
            rtt_dict = {
                "edge1": 20,
                "edge2": 40,
                "edge3": 60
            }
            
            # 如果是mine类型的文件，从rttDict获取实际RTT
            if "mine" in bandwidth_files["edge3"]:
                ip = bandwidth_files["edge3"].split("/")[-2]  # 从路径中提取IP
                rtt_dict["edge3"] = self.rttDict.get(ip, 60)  # 如果找不到IP对应的RTT，使用默认值60
            
            # 获取视频名称
            if PUREFLAF:
                video_random = random.random()
                videoName = ""
                for i in range(len(self.videoList)):
                    if video_random < self.videoList[i][1]:
                        videoName = self.videoList[i - 1][0]
                        break
                if videoName == "":
                    videoName = self.videoList[-1][0]
            else:
                if self.videoTraceIndex == len(self.videoTraceList):
                    self.videoTraceIndex = 0
                videoName = self.videoTraceList[self.videoTraceIndex].strip()  # 确保去除空白字符
            
            # 初始化客户端
            reqBI = self.client.init(
                videoName=videoName,  # 现在videoName应该是纯净的视频名称
                bandwidthFiles=bandwidth_files,
                rtt=rtt_dict,
                bwType="mixed"
            )
            
            # 初始化状态变量
            reqBitrate = BITRATES[reqBI]
            lastBitrate = 0
            buffer = 0
            hThroughput = throughput_mean
            mThroughput = throughput_mean
            server_idx = 0
            last_server_id = -1
            
            # 初始化状态向量
            state_ = [
                reqBitrate / BITRATES[-1],
                lastBitrate / BITRATES[-1],
                (buffer/1000 - 30) / 10,
                (hThroughput - throughput_mean) / throughput_std,
                (mThroughput - throughput_mean) / throughput_std,
                server_idx / 2,
                (last_server_id + 1) / 3
            ]
            
            state = state_.copy()
            
            # 开始测试------------------------------
            total_step = 0
            segNum = 0
            r_sum = 0
            
            while True:
                # 获取动作掩码
                mask = self.get_action_mask(videoName)
                
                if sum(mask) == 1:
                    a = mask.index(1)
                else:
                    if self.policy == "no_policy":
                        a = 5
                    elif self.policy == "RL":
                        # 转换状态数据
                        state_array = np.array(state)
                        state_tensor = v_wrap(state_array[None, :])
                        a, logits = self.lnet.choose_action(mask, state_tensor)
                        
                        # 调试信息（只在debug=True时输出）
                        if self.debug:
                            server_idx = a // 6
                            action_idx = a % 6
                            server_name = ["edge1", "edge2", "edge3"][server_idx]
                            bitrate = BITRATES[action_idx] if action_idx < 5 else "origin"
                            print(f"\nAction Details:")
                            print(f"Selected Server: {server_name}")
                            print(f"Selected Bitrate: {bitrate}Kbps")
                            print(f"State shape: {state_array.shape}")
                            print(f"State values: {state_array}")
                        
                        self.saveLogits(videoName, logits)
                    elif self.policy == "lower":
                        a = self.choose_action_lower(mask, reqBI)
                    elif self.policy == "closest":
                        a = self.choose_action_closest(mask, reqBI)
                    elif self.policy == "highest":
                        a = self.choose_action_highest(mask, reqBI)
                    elif self.policy == "prefetch":
                        a = self.choose_action_prefetch(mask, reqBI)
                    else:
                        print("想啥呢")
                        return

                # if random.randint(0, 1000) == 1:
                #     print("reqb=", reqBitrate, "lb=", lastBitrate, "buffer=", int(buffer), "hT=", int(hThroughput),
                #           "mT=", int(mThroughput), "busy=", round(busy, 2),
                #           "mask=", mask, "action=", a, "reqBI=", reqBI, "reward=", round(reward, 2), "logits=", logits)

                # 使用busyList
                busy = busyList[segNum % len(busyList)]
                if a == 5:
                    hitFlag = False
                else:
                    hitFlag = True

                # 获取服务器索引和具体动作
                server_idx = a // 6
                action_idx = a % 6
                
                # 使用正确的服务器名称
                server_names = ["edge1", "edge2", "edge3"]  # 使与client.py中一致的名称
                server_name = server_names[server_idx]  # 使用实际的服务器名称而不是"server_0"
                
                # 更新当前服务器
                self.current_server = server_idx
                
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = self.client.run(
                    action_idx, busy, hitFlag, server_name)  # 传入正确的服务器名称


                # 获取上一个服务器ID
                last_server_id = -1 if self.client.last_server is None else \
                                int(self.client.last_server.split('edge')[1]) - 1

                # 更新状态向量
                state_[0] = reqBitrate / BITRATES[-1]
                state_[1] = lastBitrate / BITRATES[-1]
                state_[2] = (buffer/1000 - 30) / 10
                state_[3] = (hThroughput - throughput_mean) / throughput_std
                state_[4] = (mThroughput - throughput_mean) / throughput_std
                state_[5] = server_idx / 2
                state_[6] = (last_server_id + 1) / 3

                r_sum += reward
                total_step += 1
                if done:
                    break
                state = state_.copy()
            # 结束测试------------------------------
            r_avg = r_sum / total_step if total_step > 0 else 0
            r_avg_sum += r_avg
            
            # 在循环内部记录服务器奖励
            current_server = SERVERS[a // 6]  # 使用 a 而不是 action
            server_rewards[current_server]["sum"] += reward
            server_rewards[current_server]["count"] += 1
            
            # 写入结果文件
            resFile.write(f"{r_avg}\n")
            resFile.flush()
            
            # 输出结果
            print(f"Summary: {self.bw_type} | {self.policy} | Video:{videoName} | "
                  f"Progress:{file_index}/{len(self.bandwidth_files['edge1'])} | "
                  f"Avg Reward:{r_avg:.2f}")
            
            self.videoTraceIndex += 1
        
        # 计算并打印每个服务器的统计信息
        for server, stats in server_rewards.items():
            if stats["count"] > 0:
                avg = stats["sum"] / stats["count"]
                print(f"{server} 平均奖励: {avg:.2f} (使用次数: {stats['count']})")
        
        # 计算总体平均奖励
        total_reward = sum(s["sum"] for s in server_rewards.values())
        total_count = sum(s["count"] for s in server_rewards.values())
        
        resFile.close()
        return total_reward / total_count if total_count > 0 else 0




def main():
    bwTypes = ["FCC", "HSDPA", "mine"]
    modelDir = "../data/RL_model/2024-11-24_18-12-16"
    modelName = "/model/best_model.pth"
    policys = ["RL", "no_policy", "lower", "closest", "highest", "prefetch"]
    policys = ["prefetch"]
    traceIndex = "0"
    
    # 定义每个服务器的缓存文件
    cache_files = {
        "edge1": "../file/cachedFile/cachedFile_20190612_pure.txt",
        "edge2": "../file/cachedFile/cachedFile_20190529.txt",
        "edge3": "../file/cachedFile/cachedFile_20190507.txt"
    }

    for bwType in bwTypes:
        for policy in policys:
            worker = Worker(modelDir, modelName, policy, traceIndex, cache_files, bwType)
            worker.debug = True  # 启用调试输出
            worker.run()


if __name__ == "__main__":
    main()