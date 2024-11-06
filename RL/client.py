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
REBUF_PENALTY = 3
MISS_PENALTY = 0.5
START_BUFFER_SIZE = 12000  # When buffer is larger than 4s, video start to play.
MAX_BUFFER_SIZE = 60000
MIN_BUFFER_SIZE = 5000
BITRATES = [350, 600, 1000, 2000, 3000]
M_IN_K = 1000

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
        self.model_rtt = joblib.load("../../data/throughputRelation/data/2/h2m/model/2019-06-15_16-14-27/rtt/lgbm.m")
        self.model_nortt = joblib.load("../../data/throughputRelation/data/2/h2m/model/2019-06-15_16-14-27/nortt/lgbm.m")


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


    def init(self, videoName, bandwidthFileName, rtt, bwType):
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
        self.bandwidthFileName = bandwidthFileName
        self.bandwidthList = self.getBandwidthList(self.bandwidthFileName)
        self.videoSizeList  = getChunkSizeList(self.videoName)
        # print('video name=', self.videoName)
        self.parseMPDFile(self.videoName)
        self.rtt = rtt

        if self.type == "test":
            time_now = int(time.time())
            time_local = time.localtime(time_now)
            dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
            if os.path.exists(self.traceDir) == False:
                os.makedirs(self.traceDir)
            csvFileName = self.traceDir + "/" + self.videoName + "_" + str(dt) + "_" +bandwidthFileName.split("/")[-1].split(".")[0] + ".csv"
            self.csvFile = open(csvFileName, 'w')
            self.csvFile.write("No\tcSize\tHit\tbuffer\tbI\taBI\tlastBI\tbitrate\tthroughput\thThroughput\tmThroughput\tdownloadT\trebufferT\tqoe\treward\ttime\tbusy\n")


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


    def run(self, action, busy, hitFlag):
        if self.bwEndTime == 0:
            return 0, 0, 0, 0, 0, 0, True, 0
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
        residualContentLength = self.contentLength * 8.0 # bit
        self.downloadTime = 0.0
        while residualContentLength > 0.0 :
            if self.bwIndex >= len(self.bandwidthList) - 1:
                self.bwIndex = 0
                self.bwStartTime = self.bandwidthList[0][0] # self.bandwidthList[i] = [time, bandwidth]
            relativeTime = (self.bwStartTime + self.currentTime) % self.bwEndTime
            hBandwidth = self.bandwidthList[-1][1]
            for i in range(self.bwIndex, len(self.bandwidthList)):
                if self.bandwidthList[i][0] > relativeTime:
                    hBandwidth = self.bandwidthList[i][1]
                    self.bwIndex = i
                    break
            if residualContentLength > hBandwidth:
                self.downloadTime += 1.0
                residualContentLength -= hBandwidth
                self.currentTime += 1.0
            else:
                self.downloadTime += residualContentLength/hBandwidth
                residualContentLength = 0.0
                self.currentTime += residualContentLength/hBandwidth
        # throughput and download time--------------------------------------------------
        self.hThroughput = self.contentLength * 8 / self.downloadTime

        if self.rtt == -1:
            order = ['size', 'hThroughput', 'hTime']
            input = {'size': [self.contentLength], 'hThroughput': [self.hThroughput], 'hTime': [self.downloadTime]}
            input_df = pd.DataFrame(input)
            input_df = input_df[order]
            lam=0.1
            for col in ('hTime', 'size', 'hThroughput'):
                input_df[col] = boxcox1p(input_df[col], lam)
            self.mthroughput = np.exp(self.model_nortt.predict(input_df)[0])
        else:
            order = ['size', 'hThroughput', 'hTime', 'rtt']
            input = {'size': [self.contentLength], 'hThroughput': [self.hThroughput], 'hTime': [self.downloadTime],
                     'rtt': [self.rtt]}
            input_df = pd.DataFrame(input)
            input_df = input_df[order]
            lam=0.1
            for col in ('hTime', 'size', 'rtt', 'hThroughput'):
                input_df[col] = boxcox1p(input_df[col], lam)
            self.mthroughput = np.exp(self.model_rtt.predict(input_df)[0])

        mdownloadTime = self.contentLength * 8 / self.mthroughput

        if self.hitFlag == 0:
            self.throughput = self.mthroughput
            self.downloadTime = mdownloadTime
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

        self.qoe = BITRATES[self.actualBitrateI] / M_IN_K \
              - qualityVariation / M_IN_K \
              - REBUF_PENALTY * self.rebufferTimeOneSeg

        # 直接使用 QoE 作为奖励
        self.reward = self.qoe

        # 删除或注释掉原有的奖励计算
        # if self.actualBitrateI != 0:
        #     self.reward = self.qoe - (1 - self.hitFlag) * (BITRATES[self.actualBitrateI]-BITRATES[self.actualBitrateI - 1])/ M_IN_K
        # else:
        #     self.reward = self.qoe

        # ABR ----------------------------------------------
        self.last_bitrate_index = self.actualBitrateI
        if self.bufferSize < MIN_BUFFER_SIZE:
            self.bitrateIndex = 0
        else:
            self.bitrateIndex = self.getBitrateIndex(self.throughput)
        # ABR ----------------------------------------------
        self.segmentNum = self.segmentNum + 1
        if self.segmentNum > self.segmentCount:
            done = True

        if self.bufferSize + self.segementDuration > MAX_BUFFER_SIZE:
            self.currentTime += (self.bufferSize + self.segementDuration - MAX_BUFFER_SIZE)/1000
            self.bufferSize = MAX_BUFFER_SIZE - self.segementDuration
        self.busy = busy
        # state =[lastBitrate buffer hThroughput mThroughput rtt busy mask]
        return BITRATES[self.bitrateIndex], BITRATES[self.last_bitrate_index], self.bufferSize, self.hThroughput, self.mthroughput, \
               self.reward, self.bitrateIndex, done, self.segmentNum
