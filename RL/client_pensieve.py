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
import torch
import torch.nn as nn
import torch.nn.functional as F


TURN_COUNT = 1
RUN_DURATION = 1800
# REBUF_PENALTY = 2.15
REBUF_PENALTY = 2.15
MISS_PENALTY = 0.5
START_BUFFER_SIZE = 12000  # When buffer is larger than 4s, video start to play.
MAX_BUFFER_SIZE = 60000
MIN_BUFFER_SIZE = 5000
BITRATES = [350, 600, 1000, 2000, 3000]
M_IN_K = 1000
S_INFO = 6  # 状态维度
S_LEN = 8   # 状态历史长度
A_DIM = 5   # 动作维度（码率选择）

originServers=["local","local_noproxy","remote_noproxy"]
originServer = originServers[1]

if originServer == "local":
    URLPrefix = "http://219.223.189.148:80/video"
    host = "219.223.189.148"
elif originServer == "local_noproxy":
    URLPrefix = "http://219.223.189.147:80/video"
    host = "219.223.189.147"
elif originServer == "remote_noproxy":
    URLPrefix = "http://39.106.193.51/video"
    host = "39.106.193.51"


def getChunkSizeList(videoName):
    file = open("../file/videoSize/" + videoName + ".txt")
    lines = file.readlines()
    sList = [[int(i.split(" ")[1]), int(i.split(" ")[2]), int(i.split(" ")[3]), int(i.split(" ")[4]), int(i.split(" ")[5])] for i in lines]
    return sList


def getRttList():
    file = open("../file/rtt/rtt.txt")
    lines = file.readlines()
    rttList = [float(i.split(" ")[1]) for i in lines]
    return rttList


def load_pensieve_model():
    """加载Pensieve模型"""
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            # Actor网络
            self.actor = nn.ModuleList([
                nn.Conv1d(S_INFO, 128, 4, padding=1, padding_mode='replicate'),
                nn.Conv1d(128, 128, 4, padding=1, padding_mode='replicate'),
                nn.Conv1d(128, 128, 4, padding=1, padding_mode='replicate'),
                nn.Linear(640, 128),  # 128 * 5 = 640
                nn.Linear(128, A_DIM)
            ])
            
            # Critic网络
            self.critic = nn.Sequential(
                nn.Linear(S_INFO * S_LEN, 128),  # 6 * 8 = 48
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            self._initialize_weights()
    
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0., std=0.1)
                    nn.init.constant_(m.bias, 0.)
        
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

    # 创建模型实例
    model = Net()
    
    # 加载预训练权重
    model_path = "../data/Pensieve_Model/2025-01-13_14-12-06/model/pensieve_model_best.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        print(f"Warning: No pretrained model found at {model_path}")
    
    return model


class Client:
    def __init__(self, type, traceDir = "" ):
        self.bandwidthList = []
        self.segementDuration = -1 # unit:ms
        self.bitrateSet = []
        self.bitrateIndex = -1
        self.last_bitrate_index = -1
        self.lastQoe = -1
        self.segmentCount = -1
        self.videoDuration = -1 # unit:s
        self.segmentNum = -1 #current segment index
        self.bufferSize = -1
        self.throughputList = []
        self.pDownloadTList = [] # state
        self.pThroughputList = [] # state
        self.startTime = -1
        self.startupFlag = True
        self.turnEndFlag = False
        self.currentTime = 0
        self.totalRebufferTime = -1
        self.videoName = ""
        self.currentTurn = 1
        self.cachedList = []
        self.throughputHarmonic = 0
        self.rttList = getRttList()
        self.type = type
        if self.type == "test":
            self.VIDEO_RANDOM = 1
        self.traceDir = traceDir
        self.bwIndex = -1
        # 添加多服务器相关的属性
        self.current_server = None  # 当前使用的服务器
        self.bandwidthLists = {}    # 存储不同服务器的带宽列表
        self.servers = ["edge1", "edge2", "edge3"]  # 可用的服务器列表
        # 加载throughput的预测模型
        self.model_rtt = joblib.load("../../data/throughputRelation/data/2/h2m/model/2019-06-15_16-14-27/rtt/lgbm.m")
        self.model_nortt = joblib.load("../../data/throughputRelation/data/2/h2m/model/2019-06-15_16-14-27/nortt/lgbm.m")
        # 添加 last_server 属性
        self.last_server = None
        self.current_server = None
        self.servers = ["edge1", "edge2", "edge3"]
        self.pensieve_model = load_pensieve_model()  # 加载预训练的模型

        # 添加必要的属性初始化
        self.throughput_mean = 2297514.2311790097
        self.throughput_std = 4369117.906444455
        self.busy = 0
        self.hThroughput = 0
        self.mthroughput = 0
        self.last_bitrate_index = -1

        # 初始化历史记录列表
        self.throughput_history = []
        self.download_history = []
        self.last_bitrate_index = -1
        self.segmentCount = 0
        self.segmentNum = 0
        self.videoSizeList = []
        self.bufferSize = 0


    def getBandwidthList(self, fileName, start = 0):
        file = open(fileName)
        # print(fileName)
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


    def getBitrateIndex(self, throughput):
        """使用Pensieve模型选择码率"""
        # 构建状态向量
        state = np.zeros((S_INFO, S_LEN))
        
        # 上一个码率
        state[0, -1] = self.last_bitrate_index / float(len(BITRATES)) if self.last_bitrate_index != -1 else 0
        
        # 缓冲区大小
        state[1, -1] = self.bufferSize / 1000.0  # ms -> s
        
        # 吞吐量历史
        curr_idx = S_LEN - len(self.throughput_history)
        for i, throughput in enumerate(self.throughput_history):
            state[2, curr_idx + i] = throughput / M_IN_K / M_IN_K  # bytes/ms -> Mbps
        
        # 下载时间历史
        curr_idx = S_LEN - len(self.download_history)
        for i, download in enumerate(self.download_history):
            state[3, curr_idx + i] = download
        
        # 剩余块数
        state[4, -1] = self.segmentCount - self.segmentNum
        
        # 下一个块的大小
        if self.segmentNum < len(self.videoSizeList):
            next_chunk_sizes = self.videoSizeList[self.segmentNum]
            for i in range(len(BITRATES)):
                state[5, i] = next_chunk_sizes[i]
        
        # 使用模型预测
        state_tensor = torch.FloatTensor(state)  # [S_INFO, S_LEN]
        state_tensor = state_tensor.unsqueeze(0)  # 添加batch维度 [1, S_INFO, S_LEN]
        
        with torch.no_grad():
            action_probs, _ = self.pensieve_model(state_tensor)
            action = torch.argmax(action_probs).item()
        
        return action


    def init(self, videoName, bandwidthFiles, rtt, bwType):
        self.segementDuration = -1  # unit:ms
        self.bitrateSet = []
        self.bitrateIndex = 0
        self.last_bitrate_index = -1
        self.segmentCount = -1
        self.videoDuration = -1  # unit:s

        self.segmentNum = 1  # current segment index
        self.contentLength = -1
        self.hitFlag = -1
        self.bufferSize = 0
        self.actualBitrateI = 0
        self.throughput = -1
        self.hThroughput = -1
        self.mthroughput = -1
        self.downloadTime = -1
        self.rebufferTimeOneSeg = -1
        self.qoe = -1
        self.reward = -1
        self.busy = -1

        self.throughputList = []
        self.pDownloadTList = [] # state
        self.pThroughputList = [] # state
        self.currentTime = 0
        self.startTime = 0  # 启动时间
        self.startupFlag = True
        self.totalRebufferTime = 0
        self.videoName = videoName
        self.consume_buffer_flag = False
        self.throughputHarmonic = 0
        self.bandwidthFiles = bandwidthFiles  # 存储所有edge server的带宽文件
        self.bandwidthLists = {}  # 存储所有edge server的带宽列表
        
        # 为每个edge server加载带宽列表
        for server, bw_file in bandwidthFiles.items():
            self.bandwidthLists[server] = self.getBandwidthList(bw_file)
        
        self.videoSizeList  = getChunkSizeList(self.videoName)
        # print('video name=', self.videoName)
        self.parseMPDFile(self.videoName)
        self.rtt = rtt

        if self.type == "test":
            time_now = int(time.time())
            time_local = time.localtime(time_now)
            dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
            
            # 使用edge1的带宽文件名作为CSV文件名的一部分
            bw_file_name = bandwidthFiles["edge1"].split("/")[-1].split(".")[0]
            
            if os.path.exists(self.traceDir) == False:
                os.makedirs(self.traceDir)
            
            csvFileName = self.traceDir + "/" + self.videoName + "_" + str(dt) + "_" + bw_file_name + ".csv"
            self.csvFile = open(csvFileName, 'w')
            # 保持原有的CSV header格式
            self.csvFile.write("No\tcSize\tHit\tbuffer\tbI\taBI\tlastBI\tbitrate\tthroughput\thThroughput\tmThroughput\tdownloadT\trebufferT\tqoe\treward\ttime\tbusy\n")
       
        # 返回初始bitrate index
        return self.bitrateIndex 


    def parseMPDFile(self, videoName):
        self.bitrateSet = []
        lineCount = 1
        VideoStartLineCount = -1
        AudioStartLineCount = -1
        self.segmentCount = -1
        self.videoDuration = -1

        responseStr = open("../file/video_mpd/"+videoName+"/stream.mpd").read()
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


    def run(self, action, busy, hitFlag, server_name, is_prefetch=False):
        """
        @param action: 动作索引
        @param busy: 忙碌程度
        @param hitFlag: 是否命中缓存
        @param server_name: 服务器名称
        @param is_prefetch: 是否是预取操作
        """
        # log ---------------------------------------------------------
        ss = str(self.segmentNum)+"\t"+str(self.contentLength)+"\t"+str(self.hitFlag)+"\t"+str(int(self.bufferSize))+"\t"+\
             str(self.bitrateIndex)+"\t"+str(self.actualBitrateI)+"\t"+str(self.last_bitrate_index)+"\t"+str(BITRATES[self.actualBitrateI])+"\t"+\
             str(int(self.throughput))+"\t"+str(int(self.hThroughput))+"\t"+str(int(self.mthroughput))+"\t"+\
             str(round(self.downloadTime,2))+"\t"+str(round(self.rebufferTimeOneSeg,2))+"\t"+\
             str(round(self.qoe,2))+"\t"+str(round(self.reward,2))+"\t"+str(int(self.currentTime)) + "\t" + str(self.busy)
        # print("No\tcSize\tHit\tbuffer\tbI\taBI\tlastBI\tbitrate\tthroughput\thThroughput\tmThroughput\tdownloadT\trebufferT\tqoe\treward\ttime\tbusy")
        # print(ss)
        if self.type == "test":
            self.csvFile.write(ss+"\n")
            self.csvFile.flush()
        # log ---------------------------------------------------------------
        s = []
        self.qoe = 0
        done = False

        self.hitFlag = 0
        if hitFlag:
            self.actualBitrateI = action
            self.hitFlag = 1
        else:
            self.actualBitrateI = self.bitrateIndex
            self.hitFlag = 0

        # if action == 5: # miss
        #     self.actualBitrateI = self.bitrateIndex
        # else:
        #     self.actualBitrateI = action
        #     self.hitFlag = 1


        self.contentLength = self.videoSizeList[self.segmentNum-1][self.actualBitrateI] # Byte
        # ---------------------------------------------------------------
        residualContentLength = self.contentLength * 8.0  # bit
        self.downloadTime = 0.0
        
        # 更新当前服务器
        self.last_server = self.current_server
        self.current_server = server_name
        
        # 使用对应服务器的带宽列表
        # if hitFlag:
        if server_name in ["edge2", "edge3"]:
            # 计算从edge2/3到edge1的传输时间
            edge_to_edge_time = self.calculate_transfer_time(
                residualContentLength,
                self.bandwidthLists[server_name],
                self.bwIndex
            )
            
            # 计算从edge1到client的传输时间
            edge1_to_client_time = self.calculate_transfer_time(
                residualContentLength,
                self.bandwidthLists["edge1"],
                self.bwIndex
            )
            
            self.downloadTime = edge_to_edge_time + edge1_to_client_time
        else:
            # edge1直接到client的传输时间
            self.downloadTime = self.calculate_transfer_time(
                residualContentLength,
                self.bandwidthLists["edge1"],
                self.bwIndex
            )
            
        # 如果是prefetch操作，增加下载时间
        if is_prefetch:
            self.downloadTime *= 2
            
        self.hThroughput = self.contentLength * 8 / self.downloadTime
        
        

        if self.rtt == -1:
            # 使用不带 RTT 的模型
            order = ['size', 'hThroughput', 'hTime']
            input = {
                'size': [self.contentLength], 
                'hThroughput': [self.hThroughput], 
                'hTime': [self.downloadTime]
            }
        else:
            # 使用带 RTT 的模型，使用当前服务器的 RTT 值
            current_rtt = self.rtt.get(server_name, -1)  # 获取当前服务器的 RTT 值
            order = ['size', 'hThroughput', 'hTime', 'rtt']
            input = {
                'size': [self.contentLength], 
                'hThroughput': [self.hThroughput], 
                'hTime': [self.downloadTime],
                'rtt': [current_rtt]  # 使用当前服务器的 RTT 值
            }
        
        input_df = pd.DataFrame(input)
        input_df = input_df[order]
        lam = 0.1
        
        # 对每列进行数据处理
        for col in input_df.columns:
            # 应用 Box-Cox 转换
            input_df[col] = boxcox1p(input_df[col], lam)
        
        # 根据模型类型预测
        if self.rtt == -1:
            self.mthroughput = np.exp(self.model_nortt.predict(input_df)[0])
        else:
            self.mthroughput = np.exp(self.model_rtt.predict(input_df)[0])

        #mdownloadTime = self.contentLength * 8 / self.mthroughput   # 预测的下载时间
        mdownloadTime = self.contentLength * 8 / self.mthroughput + self.contentLength * 8 / self.hThroughput  # 预测的下载时间加上回原站取视频的时间开销

        if self.hitFlag == 0:
            self.throughput = self.mthroughput
            self.downloadTime = mdownloadTime
            if is_prefetch:
                self.downloadTime *= 2
        else:
            self.throughput = self.hThroughput
        # buffer size and rebuffer--------------------------------------------------
        self.rebufferTimeOneSeg = -1
        if self.startupFlag:
            self.startTime += self.downloadTime
            self.rebufferTimeOneSeg = 0 #downloadTime
        else:
            if self.downloadTime * 1000 > self.bufferSize:
                self.rebufferTimeOneSeg = self.downloadTime - self.bufferSize / 1000
                if self.rebufferTimeOneSeg > 3:
                    self.rebufferTimeOneSeg = 3
                self.bufferSize = 0
                self.totalRebufferTime += self.rebufferTimeOneSeg
            else:
                self.bufferSize = self.bufferSize - self.downloadTime * 1000
                self.rebufferTimeOneSeg = 0
        self.bufferSize += self.segementDuration
        if self.bufferSize > MIN_BUFFER_SIZE:
            self.startupFlag = False
        # ---------------------------------------------------------------
        if len(self.pDownloadTList) == 0:
            self.pDownloadTList = [self.downloadTime]*5
            self.pThroughputList = [self.throughput]*5
        else:
            self.pDownloadTList = self.pDownloadTList[1:5]+[self.downloadTime]
            self.pThroughputList = self.pThroughputList[1:5] + [self.throughput]
        # qoe ---------------------------------------------------------------
        if self.last_bitrate_index == -1:
            qualityVariation = BITRATES[self.actualBitrateI]
        else:
            qualityVariation = abs(BITRATES[self.actualBitrateI] - BITRATES[self.last_bitrate_index])

        self.qoe = BITRATES[self.actualBitrateI]  / M_IN_K \
              - qualityVariation  / M_IN_K \
              - REBUF_PENALTY * self.rebufferTimeOneSeg
        # if self.actualBitrateI != 0:
        #     self.reward =  self.qoe - (1 - self.hitFlag) * (BITRATES[self.actualBitrateI]-BITRATES[self.actualBitrateI - 1])/ M_IN_K
        # else:
        self.reward = self.qoe
        # ABR ----------------------------------------------
        self.last_bitrate_index = self.actualBitrateI
        if self.bufferSize < MIN_BUFFER_SIZE:
            self.bitrateIndex = 0
            #print("self.bufferSize < MIN_BUFFER_SIZE")
        else:
            self.bitrateIndex = self.getBitrateIndex(self.throughput)
            #print(f"After - bitrateIndex: {self.bitrateIndex}")
        # ABR ----------------------------------------------
        self.segmentNum = self.segmentNum + 1
        if self.segmentNum > self.segmentCount:
            done = True

        if self.bufferSize + self.segementDuration > MAX_BUFFER_SIZE:
            self.currentTime += (self.bufferSize + self.segementDuration - MAX_BUFFER_SIZE)/1000
            self.bufferSize = MAX_BUFFER_SIZE - self.segementDuration
        self.busy = busy

        # 添加调试信息
        # print(f"Download Time: {self.downloadTime:.2f}s")
        # print(f"Rebuffer Time: {self.rebufferTimeOneSeg:.2f}s")
        # print(f"Selected Bitrate: {BITRATES[self.actualBitrateI]}Kbps")
        # print(f"Quality Variation: {qualityVariation}Kbps")
        # print(f"QoE Components:")
        # print(f"  Bitrate Reward: {BITRATES[self.actualBitrateI] / M_IN_K:.2f}")
        # print(f"  Quality Penalty: {qualityVariation / M_IN_K:.2f}")
        # print(f"  Rebuffer Penalty: {REBUF_PENALTY * self.rebufferTimeOneSeg:.2f}")
        # print(f"Final QoE: {self.qoe:.2f}")
        # state =[lastBitrate buffer hThroughput mThroughput rtt busy mask]

        return BITRATES[self.bitrateIndex], BITRATES[self.last_bitrate_index], self.bufferSize, self.hThroughput, self.mthroughput, \
            self.reward, self.bitrateIndex, done, self.segmentNum

    def calculate_transfer_time(self, residualContentLength, bandwidthList, bwIndex):
        "计算单端时间"
        downloadTime = 0.0
        while residualContentLength > 0.0:
            if bwIndex >= len(bandwidthList) - 1:
                bwIndex = 0
                self.bwStartTime = bandwidthList[0][0]
            relativeTime = (self.bwStartTime + self.currentTime) % self.bwEndTime
            hBandwidth = bandwidthList[-1][1]
            
            for i in range(bwIndex, len(bandwidthList)):
                if bandwidthList[i][0] > relativeTime:
                    hBandwidth = bandwidthList[i][1]
                    bwIndex = i
                    break
                    
            if residualContentLength > hBandwidth:
                downloadTime += 1.0
                residualContentLength -= hBandwidth
                self.currentTime += 1.0
            else:
                downloadTime += residualContentLength/hBandwidth
                residualContentLength = 0.0
                self.currentTime += residualContentLength/hBandwidth
                
        return downloadTime

    def preprocess_state(self, last_quality, buffer_size, throughput_history, download_history):
        # 按照Pensieve的要求处理状态
        state = np.zeros((6, S_INFO, S_LEN))
        state[0, -1] = last_quality  # last quality
        state[1, -1] = buffer_size  # buffer size
        state[2, :len(throughput_history)] = throughput_history  # throughput
        state[3, :len(download_history)] = download_history  # download time
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP)  # chunks left
        state[5, -1] = buffer_size  # buffer size
        return state
