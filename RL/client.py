#from __future__ import with_statement
# -*- coding: utf-8 -*

import math
import os
import time
import sys
import random
import math
import numpy as np
from xgboost import XGBRegressor
from scipy.special import boxcox1p
import joblib
import pandas as pd


TURN_COUNT = 1
RUN_DURATION = 1800
# REBUF_PENALTY = 2.15
REBUF_PENALTY = 1.0
MISS_PENALTY = 0.5
START_BUFFER_SIZE = 12000  # When buffer is larger than 4s, video start to play.
MAX_BUFFER_SIZE = 60000
MIN_BUFFER_SIZE = 5000
BITRATES = [350, 600, 1000, 2000, 3000]
M_IN_K = 1000

# originServers=["local","local_noproxy","remote_noproxy"]
originServers = ["local","edge1", "edge2", "edge3", "origin"]
# originServer = originServers[1]
# if originServer == "local":
#     URLPrefix = "http://219.223.189.148:80/video"
#     host = "219.223.189.148"
# elif originServer == "local_noproxy":
#     URLPrefix = "http://219.223.189.147:80/video"
#     host = "219.223.189.147"
# elif originServer == "remote_noproxy":
#     URLPrefix = "http://39.106.193.51/video"
#     host = "39.106.193.51"
EDGE_SERVERS = {
    "edge1": {
        "url": "http://edge1_ip:80/video",
        "host": "edge1_ip"
    },
    "edge2": {
        "url": "http://edge2_ip:80/video", 
        "host": "edge2_ip"
    },
    "edge3": {
        "url": "http://edge3_ip:80/video",
        "host": "edge3_ip"
    },
    "origin": {
        "url": "http://origin_ip:80/video",
        "host": "origin_ip"
    }
}
# add 3 edge servers

class Client:
    def __init__(self, type, traceDir=""):
        # 基础属性
        self.type = type
        self.traceDir = traceDir
        
        # 视频相关
        self.segementDuration = -1
        self.bitrateSet = []
        self.bitrateIndex = -1
        self.last_bitrate_index = -1
        self.segmentCount = -1
        self.videoDuration = -1
        self.segmentNum = -1
        self.videoName = ""
        self.videoSizeList = []
        
        # 缓冲区相关
        self.bufferSize = 0  # 初始buffer为0
        self.startupFlag = True
        self.totalRebufferTime = 0
        self.rebufferTimeOneSeg = 0
        self.segementDuration = 1000  # 1秒, 单位ms
        
        # 历史数据
        self.K = 5  # 历史数据跨度
        self.dict = {
            'throughputList_k': [],
            'downloadTime_k': [],
            'chunkSize_k': [],
            'ifHit_k': [],
            'buffer_k': [],
            'lastQoE': [],
            'bitrate': [],
            'rtt': [],
            'chainLength': [],
            'video_type': [],
            'bitrate_set': []
        }
        
        # 吞吐量相关
        self.throughput = 0
        self.hThroughput = 0
        self.mthroughput = 0
        self.downloadTime = 0
        self.contentLength = 0
        
        # 服务器相关
        self.current_server = "edge1"
        self.last_server = None
        self.hitFlag = 1
        self.busy = 0
        
        # 加载预测模型
        self.models = {}
        base_path = "../../data/throughputRelation/data/2/h2m/model/2019-06-15_16-14-27"
        
        for server in ["edge1", "edge2", "edge3"]:
            self.models[server] = {
                "rtt": None,
                "nortt": None
            }
            try:
                # 加载RTT模型
                rtt_path = os.path.join(base_path, "rtt", "lgbm.m")
                if os.path.exists(rtt_path):
                    self.models[server]["rtt"] = joblib.load(rtt_path)
                
                # 加载NoRTT模型
                nortt_path = os.path.join(base_path, "nortt", "lgbm.m")
                if os.path.exists(nortt_path):
                    self.models[server]["nortt"] = joblib.load(nortt_path)
                    
            except Exception as e:
                print(f"Error loading models for {server}: {e}")
        
        # QoE相关
        self.qoe = 0
        self.reward = 0
        self.lastQoe = 0
        self.currentTime = 0
        self.startTime = -1
        
        
        # QoE参数
        self.REBUF_PENALTY = 4.3  # 重缓冲惩罚系数
        self.SMOOTH_PENALTY = 1.0  # 平滑惩罚系数
        self.MISS_PENALTY = 0.5   # 未命中惩罚

        
        
        # 加载视频信息
        self.video_info = {}  # 存储视频名称和类型的映射
        try:
            video_file_path = "../file/video.txt"  # 使用正确的相对路径
            with open(video_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            video_name = parts[0]
                            video_type = int(float(parts[2]))
                            self.video_info[video_name] = video_type
            # print(f"Successfully loaded video info from {video_file_path}")
        except Exception as e:
            print(f"Error loading video info: {e}")

    def getBandwidthList(self, server_name, fileName, start=0):
        """根据不同edge server获取对应的带宽数据"""
        try:
            with open(fileName) as file:
                bList = file.readlines()
                bList = [[int(i.split(" ")[0]), float(i.split(" ")[1])] for i in bList]
                bList = [i for i in bList if i[1] > 0]
                
                if self.type == "train":
                    self.bwIndex = random.randint(0, len(bList)-10)
                else:
                    self.bwIndex = start
                    
                self.bwStartTime = bList[self.bwIndex][0]
                self.bwEndTime = bList[-1][0]
                return bList
        except FileNotFoundError:
            print(f"Error: Bandwidth file not found: {fileName}")
            print(f"Current working directory: {os.getcwd()}")
            # 返回默认带宽据
            return [[0, 1000000]] * 100  # 默认带宽为1Mbps
      
    
    

    def getBitrateIndex(self, throughput):
        if len(self.throughputList) < 5:
            self.throughputList.append(throughput)
        else:
            self.throughputList.append(throughput)
            self.throughputList.pop(0)

        reciprocal = 0
        for i in range(len(self.throughputList)):
            reciprocal += 1 / self.throughputList[i]
        reciprocal /= len(self.throughputList)

        if reciprocal != 0:
            self.throughputHarmonic = 1 / reciprocal
        else:
            self.throughputHarmonic = 0

        # print("throughput harmonic: %f" % throughputHarmonic)

        for i in range(len(self.bitrateSet)):
            if self.throughputHarmonic < self.bitrateSet[i]:
                if i - 1 < 0:
                    return i
                else:
                    return i - 1

        return len(self.bitrateSet) - 1


    def getVideoSizeList(self, videoName):
        """获取视频分片大小列表"""
        try:
            # 使用相对路径
            # 假设当前代码在 RL/ 目录下运行
            base_path = "../file/videoSize"
            video_size_path = os.path.join(base_path, f"{videoName}.txt")
            
            if not os.path.exists(video_size_path):
                # 如果指定视频不存在，使用accident.txt
                video_size_path = os.path.join(base_path, "accident.txt")
                if not os.path.exists(video_size_path):
                    print(f"Neither {videoName}.txt nor accident.txt found")
                    return self.getDefaultVideoSizeList()
                print(f"Using accident.txt instead of {videoName}.txt")
            
            size_list = []
            with open(video_size_path, 'r') as f:
                for line in f:
                    # 跳过空行
                    if not line.strip():
                        continue
                    
                    # 解析行数据
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 6:  # 序号 + 5个码率
                            sizes = [int(size) for size in parts[-5:]]
                            size_list.append(sizes)
                    except (ValueError, IndexError):
                        continue
                    
            if size_list:
                return size_list
            else:
                print(f"No valid size data found in {video_size_path}")
                return self.getDefaultVideoSizeList()
            
        except Exception as e:
            print(f"Error reading video size file: {e}")
            return self.getDefaultVideoSizeList()

    def getDefaultVideoSizeList(self):
        """创建默认的视频大小列表"""
        try:
            # 尝试从accident.txt读取典型值
            accident_path = os.path.join("../file/videoSize", "accident.txt")
            if os.path.exists(accident_path):
                with open(accident_path, 'r') as f:
                    # 读取前10行来获取典型值
                    typical_sizes = []
                    for _ in range(10):
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 6:
                                sizes = [int(size) for size in parts[-5:]]
                                typical_sizes.append(sizes)
                
                    if typical_sizes:
                        # 计算平均值
                        avg_sizes = [
                            sum(size[i] for size in typical_sizes) // len(typical_sizes)
                            for i in range(5)
                        ]
                        return [avg_sizes for _ in range(200)]
        except Exception:
            pass
        
        # 使用硬编码的默认值
        return [
            [160000, 260000, 420000, 850000, 1200000]
            for _ in range(200)
        ]

    def init(self, videoName, bandwidthFiles, rtt, bwType):
        """初始化客户端"""
        self.videoName = videoName
        
        # 设置video_type（只从video.txt读取一次）
        if videoName in self.video_info:
            self.video_type = self.video_info[videoName]
        else:
            self.video_type = 1
        
        self.bwType = bwType
        self.rtt = rtt
        
        # 获取视频大小列表
        self.videoSizeList = self.getChunkSizeList(videoName)
        if not self.videoSizeList:
            print(f"Failed to get video size list for {videoName}")
            return None
        
        # 解析MPD文件获取视频信息
        self.parseMPDFile(videoName)
        
        # 初始化视频相关属性
        self.segmentNum = 0
        self.bitrateIndex = 0
        self.last_bitrate_index = -1
        self.actualBitrateI = 0
        self.segementDuration = 4000  # 4秒一个分片
        
        # 初始化缓冲区
        self.bufferSize = 0
        self.startupFlag = True
        self.totalRebufferTime = 0
        self.rebufferTimeOneSeg = 0
        
        # 加载带宽数据
        self.bandwidth_data = {}
        for server_name, bw_file in bandwidthFiles.items():
            try:
                self.bandwidth_data[server_name] = self.getBandwidthList(server_name, bw_file)
            except Exception as e:
                print(f"Failed to load bandwidth data for {server_name}: {e}")
                self.bandwidth_data[server_name] = []
        
        if self.type == "test":
            try:
                time_now = int(time.time())
                time_local = time.localtime(time_now)
                dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
                
                if not os.path.exists(self.traceDir):
                    os.makedirs(self.traceDir)
                
                # 使用相同的命名格式
                first_bw_file = next(iter(bandwidthFiles.values())).split("/")[-1].split(".")[0]
                self.csvFileName = f"{self.traceDir}/{videoName}_{dt}_{first_bw_file}.csv"
                self.csvFile = open(self.csvFileName, 'w')
                
                # 使用原始的表头
                headers = [
                    "No", "cSize", "Hit", "buffer", "bI", "aBI", "lastBI",
                    "bitrate", "throughput", "hThroughput", "mThroughput",
                    "downloadT", "rebufferT", "qoe", "reward", "time", "busy"
                ]
                self.csvFile.write("\t".join(headers) + "\n")
                
                # 写入初始状态行
                initial_state = ["-1"] * len(headers)
                initial_state[0] = "1"  # No
                initial_state[3] = "0"  # buffer
                initial_state[4] = "0"  # bI
                initial_state[5] = "0"  # aBI
                initial_state[7] = str(BITRATES[0])  # bitrate
                initial_state[15] = "0"  # time
                self.csvFile.write("\t".join(initial_state) + "\n")
                
            except Exception as e:
                print(f"Error creating trace file: {e}")
        
        return self.bitrateIndex

    def getDefaultBandwidthList(self):
        """返回默认的带宽数据"""
        return [[i, 1000000] for i in range(100)]  # 默认1Mbps带宽

    def parseMPDFile(self, videoName):
        self.bitrateSet = []
        lineCount = 1
        VideoStartLineCount = -1
        AudioStartLineCount = -1
        self.segmentCount = -1
        self.videoDuration = -1

        responseStr = open("../file/video_mpd/"+ videoName.split()[0]+"/stream.mpd").read()
        lines = responseStr.split('\n')
        for line in lines:
            if line.find("MPD mediaPresentationDuration")!=-1:
                mediaPresentationDuration = line.split('"')[1]
                mediaPresentationDuration = mediaPresentationDuration[2:len(mediaPresentationDuration)]
                if mediaPresentationDuration.find("H") != -1 :
                    mediaPresentationDuration_hour = int(mediaPresentationDuration.split("H")[0])
                    mediaPresentationDuration_minute = int(mediaPresentationDuration.split("H")[1].split("M")[0])
                    mediaPresentationDuration_second = float(mediaPresentationDuration.split("H")[1].split("M")[1].split("S")[0])
                    self.videoDuration = mediaPresentationDuration_hour * 3600 + mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second
                elif mediaPresentationDuration.find("M")!= -1:
                    mediaPresentationDuration_minute = int(mediaPresentationDuration.split("M")[0])
                    mediaPresentationDuration_second = float(mediaPresentationDuration.split("M")[1].split("S")[0])
                    self.videoDuration = mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second

                else:
                    mediaPresentationDuration_second = float(mediaPresentationDuration.split("S")[0])
                    self.videoDuration = mediaPresentationDuration_second

            if line.find("Video")!=-1:
                VideoStartLineCount = lineCount
            if line.find("Audio")!=-1:
                AudioStartLineCount = lineCount
            if line.find('<SegmentTemplate')!=-1 and AudioStartLineCount == -1:
                elements = line.split(' ')
                for element in elements:
                    if element.startswith("duration"):
                        self.segementDuration = int(element.split('"')[1])
            if line.find('<Representation')!=-1 and AudioStartLineCount == -1:
                elements = line.split(' ')
                for element in elements:
                    if element.startswith("bandwidth"):
                        self.bitrateSet.append(int(element.split('"')[1]))
        self.segmentCount =math.ceil(self.videoDuration / self.segementDuration * 1000)
        # print('segmentCount: %d' %self.segmentCount)
        return True

    def update_history(self):
        """更新历史数据"""
        if len(self.dict['downloadTime_k']) < self.K:
            self.dict['downloadTime_k'] = [self.downloadTime] * self.K
            self.dict['chunkSize_k'] = [self.contentLength] * self.K
            self.dict['throughputList_k'] = [self.throughput] * self.K
            self.dict['ifHit_k'] = [self.hitFlag] * self.K
            self.dict['buffer_k'] = [self.bufferSize] * self.K
        else:
            self.dict['downloadTime_k'] = self.dict['downloadTime_k'][1:] + [self.downloadTime]
            self.dict['chunkSize_k'] = self.dict['chunkSize_k'][1:] + [self.contentLength]
            self.dict['throughputList_k'] = self.dict['throughputList_k'][1:] + [self.throughput]
            self.dict['ifHit_k'] = self.dict['ifHit_k'][1:] + [self.hitFlag]
            self.dict['buffer_k'] = self.dict['buffer_k'][1:] + [self.bufferSize]

    def predict_throughput(self, server_name, current_rtt):
        """预测吞吐量"""
        try:
            # 准备输入数据
            input_data = {
                'size': [self.contentLength],
                'hThroughput': [self.hThroughput],
                'hTime': [self.downloadTime]
            }
            
            if current_rtt != -1:
                input_data['rtt'] = [current_rtt]
                order = ['size', 'hThroughput', 'hTime', 'rtt']
                model = self.models[server_name]["rtt"]
            else:
                order = ['size', 'hThroughput', 'hTime']
                model = self.models[server_name]["nortt"]
                
            if model is None:
                return self.hThroughput
                
            # 准备DataFrame
            input_df = pd.DataFrame(input_data)[order]
            
            # 数据转换
            lam = 0.1
            for col in order:
                input_df[col] = boxcox1p(input_df[col], lam)
            
            # 预测
            return np.exp(model.predict(input_df)[0])
            
        except Exception as e:
            print(f"Prediction error for {server_name}: {e}")
            return self.hThroughput

    def calculate_qoe(self):
        """计算QoE，使用码率而不是SSIM"""
        try:
            # 计算quality variation (码率变化惩罚)
            if self.last_bitrate_index == -1:
                quality_variation = BITRATES[self.actualBitrateI]
            else:
                quality_variation = abs(
                    BITRATES[self.last_bitrate_index] - 
                    BITRATES[self.actualBitrateI]
                )
            
            # 增加码率奖励的权重
            bitrate_reward = BITRATES[self.actualBitrateI] / M_IN_K * 1.5  
            quality_penalty = quality_variation / M_IN_K * 1.0
            rebuffer_penalty = REBUF_PENALTY * self.rebufferTimeOneSeg * 0.5
             
            self.qoe = bitrate_reward - quality_penalty - rebuffer_penalty
            
            # 直接使用QoE作为reward
            self.reward = self.qoe
            
        except Exception as e:
            print(f"Error in calculate_qoe: {e}")
            self.qoe = 0
            self.reward = 0

    def run(self, action, busy, hitFlag, server_name):
        """运行一步仿真"""
        # 记录server切换
        self.last_server = self.current_server
        self.current_server = server_name
        
        # 设置码率和缓存命中标志
        self.hitFlag = hitFlag
        if action == 5:  # miss
            self.actualBitrateI = self.bitrateIndex
            self.hitFlag = 0
        else:
            self.actualBitrateI = action
            self.hitFlag = 1
        
        # 计算下载时间和吞吐量
        self.contentLength = self.videoSizeList[self.segmentNum][self.actualBitrateI]  # Byte
        residualContentLength = self.contentLength * 8.0  # bit
        self.downloadTime = 0.0
        
        # 预测吞吐量
        current_rtt = self.rtt.get(server_name, -1)
        self.mthroughput = self.predict_throughput(server_name, current_rtt)
        
        # 计算实际下载时间和吞吐量
        if self.hitFlag == 0:
            # 缓存未命中，使用预测的吞吐量
            self.throughput = self.mthroughput
            if self.throughput > 0:
                self.downloadTime = residualContentLength / self.throughput
                # 计算实际的调和平均吞吐量
                self.hThroughput = self.contentLength * 8 / self.downloadTime
            else:
                self.downloadTime = 0
                self.hThroughput = 0
        else:
            # 缓存命中，使用历史吞吐量
            if self.hThroughput > 0:
                self.throughput = self.hThroughput
                self.downloadTime = residualContentLength / self.throughput
            else:
                # 如果没有历史吞吐量，使用预测值
                self.throughput = self.mthroughput
                if self.throughput > 0:
                    self.downloadTime = residualContentLength / self.throughput
                    self.hThroughput = self.contentLength * 8 / self.downloadTime
                else:
                    self.downloadTime = 0
                    self.hThroughput = 0
        
        # buffer管理部分
        self.rebufferTimeOneSeg = 0
        if self.startupFlag:
            # 启动阶段：只累积buffer，不消耗
            self.currentTime += self.downloadTime
            self.rebufferTimeOneSeg = 0
        else:
            # 正常播放阶段：计算buffer消耗和rebuffer
            if self.downloadTime * 1000 > self.bufferSize:
                # buffer不足，发生rebuffer
                rebuffer_time = (self.downloadTime * 1000 - self.bufferSize) / 1000
                self.rebufferTimeOneSeg = min(rebuffer_time, 3.0)  # 限制最大rebuffer时间
                self.bufferSize = 0
                self.totalRebufferTime += self.rebufferTimeOneSeg * 1000
            else:
                # buffer充足，正常播放并消耗buffer
                self.bufferSize = self.bufferSize - self.downloadTime * 1000
                self.rebufferTimeOneSeg = 0
        
        # 添加新的segment到buffer
        self.bufferSize += self.segementDuration
        
        # buffer上限控制
        if self.bufferSize > MAX_BUFFER_SIZE:
            skip_time = (self.bufferSize - MAX_BUFFER_SIZE) / 1000
            self.currentTime += skip_time
            self.bufferSize = MAX_BUFFER_SIZE
        
        # 更新startup状态
        if self.bufferSize > MIN_BUFFER_SIZE:
            if self.startupFlag:
                # print(f"Video playback started at {self.currentTime:.2f}s with buffer {self.bufferSize/1000:.2f}s")
               self.startupFlag = False
        
        # 计算QoE
        self.calculate_qoe()
        
        # 更新历史数据
        self.update_history()
        
        # 更新状态
        self.busy = busy
        self.last_bitrate_index = self.actualBitrateI
        self.segmentNum += 1
        
        done = self.segmentNum >= self.segmentCount
        
        # 记录trace
        if self.type == "test" and hasattr(self, 'csvFile'):
            try:
                trace_data = [
                    str(self.segmentNum),                    # No
                    str(self.contentLength),                 # cSize
                    str(self.hitFlag),                      # Hit
                    str(int(self.bufferSize)),              # buffer
                    str(self.bitrateIndex),                 # bI
                    str(self.actualBitrateI),               # aBI
                    str(self.last_bitrate_index),           # lastBI
                    str(BITRATES[self.actualBitrateI]),     # bitrate
                    str(int(self.throughput)),              # throughput
                    str(int(self.hThroughput)),             # hThroughput
                    str(int(self.mthroughput)),             # mThroughput
                    f"{self.downloadTime:.2f}",             # downloadT
                    f"{self.rebufferTimeOneSeg:.2f}",       # rebufferT
                    f"{self.qoe:.2f}",                      # qoe
                    f"{self.reward:.2f}",                   # reward
                    str(int(self.currentTime)),             # time
                    f"{self.busy:.1f}"                      # busy
                ]
                
                self.csvFile.write("\t".join(trace_data) + "\n")
                self.csvFile.flush()
                
            except Exception as e:
                print(f"Error writing trace: {e}")
        
        return (BITRATES[self.bitrateIndex], BITRATES[self.last_bitrate_index],
                self.bufferSize, self.hThroughput, self.mthroughput,
                self.reward, self.bitrateIndex, done, self.segmentNum)

    def getChunkSizeList(self, videoName):
        """获取视频块大小列表"""
        try:
            # 只取视频名称的第一部分，忽略概率值
            pure_video_name = videoName.split()[0]
            video_size_path = f"../file/videoSize/{pure_video_name}.txt"
            
            if not os.path.exists(video_size_path):
                # 如果指定视频不存在，使用默认视频
                video_size_path = "../file/videoSize/accident.txt"
                if not os.path.exists(video_size_path):
                    return self.getDefaultVideoSizeList()
                
                # 记录日志
                os.makedirs("../logs", exist_ok=True)
                with open("../logs/video_substitutions.log", "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Using accident.txt instead of {pure_video_name}.txt\n")
            
            with open(video_size_path, 'r') as file:
                lines = file.readlines()
                size_list = []
                for line in lines:
                    sizes = [int(size) for size in line.strip().split()[1:6]]
                    size_list.append(sizes)
                return size_list
                
        except Exception as e:
            print(f"Error reading video size file: {e}")
            return self.getDefaultVideoSizeList()
