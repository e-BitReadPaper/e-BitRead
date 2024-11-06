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
S_LEN = 5
A_DIM = 6
PUREFLAF = False
throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455


def takeSecond(elem):
    return elem[1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #actor
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.linear_2_a = nn.Linear(200, 100)
        self.output_a = nn.Linear(100, A_DIM)

        #critic
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.linear_2_c = nn.Linear(200, 100)
        self.output_c = nn.Linear(100, 1)

        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                  self.linear_1_c, self.linear_2_c, self.output_c])
        self.distribution = torch.distributions.Categorical


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
        logits, _ = self.forward(s)
        mask_b = torch.tensor([mask]).byte()
        masked_logits = torch.masked_select(logits, mask_b)
        re = int(masked_logits.argmax())
        counter = 0
        for i in range(len(mask)):
            if mask[i] == 1:
                counter += 1
            if counter == re + 1:
                break

        return np.int64(i), logits.data.numpy()[0]


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class Worker(mp.Process):
    def __init__(self, modelDir, modelName, policy, traceIndex, cacheFile, bwType):
        super(Worker, self).__init__()
        self.modelDir = modelDir
        self.lnet = Net()
        if policy == "RL":
            self.lnet.load_state_dict(torch.load(modelDir + modelName))
        self.policy = policy
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
        self.traceIndex = traceIndex

        # self.client = Client("test", modelDir + "/trace_" + str(traceIndex) + "/" + policy + "/" + dt)
        self.client = Client("test", modelDir + "/test_trace_" + str(traceIndex) + "/" + bwType + "/" + policy)
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]
        videoTraceFile = open("../file/videoTrace.txt")
        self.videoTraceIndex = 0
        self.videoTraceList = videoTraceFile.readlines()
        self.videoTraceList = [i.strip() for i in self.videoTraceList]
        self.cachedVideo = cacheFile
        self.cachedList = []
        self.cachedList = [i.strip() for i in open(cacheFile).readlines()]
        self.cachedList = [i.split(" ")[0]+"_"+i.split(" ")[1] for i in self.cachedList]
        self.cachedVideo = []
        self.cachedVideo = [i.split("_")[0] for i in self.cachedList]
        self.bwType = bwType

        self.busyTraceL = os.listdir("../../data/trace/busy/2")
        self.rttDict = {}
        get_rttDict(self.rttDict)
        self.bandwidth_fileList = []
        self.logitsDict = {} # {video_bitrate:[logits,counter]}


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
        fileName = self.busyTraceL[random.randint(0, len(self.busyTraceL) - 1)]
        return np.loadtxt("../../data/trace/busy/2/" + fileName).flatten().tolist()


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
        for i in range(5):
            key = videoName + "_" + str(i+1)
            if key not in self.logitsDict:
                self.logitsDict[key] = logits[i]
            else:
                self.logitsDict[key] += logits[i]


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
        self.getBandwidthFileList()
        # print("bandwidth file count=", len(self.bandwidth_fileList))
        r_avg_sum = 0
        resDir = self.modelDir + "/test_res_" + str(self.traceIndex)
        if os.path.exists(resDir) == False:
            os.makedirs(resDir)
        resFile = open(resDir + "/" + self.policy + "_" + self.bwType + ".txt",'w')
        for bandwidth_file in self.bandwidth_fileList:
            bandwidth_fileName = bandwidth_file[0]
            rtt = bandwidth_file[1]
            bwType = bandwidth_file[2]
            busyList = self.get_busyTrace()
            # get video -----------------------------
            videoName = ""
            if PUREFLAF:
                video_random = random.random()
                for i in range(len(self.videoList)):
                    if video_random < self.videoList[i][1]:
                        videoName = self.videoList[i - 1][0]
                        break
                if videoName == "":
                    videoName = self.videoList[-1][0]
            else:
                if self.videoTraceIndex == len(self.videoTraceList):
                    self.videoTraceIndex = 0
                videoName = self.videoTraceList[self.videoTraceIndex]

            # print(videoName)
            # get video ----------------------------
            reqBI = self.client.init(videoName=videoName, bandwidthFileName=bandwidth_fileName, rtt=rtt, bwType=bwType)
            # mask---------------------------
            mask = [1] * A_DIM
            for bIndex in range(5):
                if PUREFLAF:
                    mask[bIndex] = 0
                else:
                    if videoName + "_" + str(bIndex + 1) not in self.cachedList:
                        mask[bIndex] = 0.
            # mask---------------------------
            state_ = np.zeros(S_LEN)
            state = state_.copy() #state =[lastBitrate buffer hThroughput mThroughput busy mask]
            # 开始测试------------------------------
            total_step = 0
            segNum = 0
            r_sum = 0
            reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum, busy = [0] * 10
            while True:
                if sum(mask) == 1:
                    a = mask.index(1)
                else:
                    if self.policy == "no_policy":
                        a = 5
                    elif self.policy == "RL":
                        a, logits = self.lnet.choose_action(mask, v_wrap(state[None, :]))
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

                busy = busyList[segNum % len(busyList)]
                if a == 5:
                    hitFlag = False
                else:
                    hitFlag = True

                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = self.client.run(a, busy, hitFlag)

                state_[0] = reqBitrate / BITRATES[-1]
                state_[1] = lastBitrate / BITRATES[-1]
                state_[2] = (buffer/1000 - 30) / 10
                state_[3] = (hThroughput - throughput_mean) / throughput_std
                state_[4] = (mThroughput - throughput_mean) / throughput_std
                reward = reward / 5

                r_sum += reward
                total_step += 1
                if done:
                    break
                state = state_.copy()
            # 结束测试------------------------------
            r_avg = r_sum / total_step
            r_avg_sum += r_avg
            resFile.write(str(r_avg) + "\n")
            resFile.flush()
            print(self.bwType, self.policy, videoName, self.bandwidth_fileList.index(bandwidth_file),"/",len(self.bandwidth_fileList), r_avg)
            self.videoTraceIndex += 1




def main():
    bwTypes = ["FCC", "HSDPA", "mine"]
    modelDir = "../../data/RL_model/2019-06-28_10-18-56"
    modelName = "/model/233293.pkl"
    policys = ["RL", "no_policy", "lower", "closest", "highest", "prefetch"]
    policys = ["RL"]
    traceIndex = "0"
    cacheFile = "../file/cachedFile/cachedFile_20190612_pure.txt"

    # test 所有方法
    '''
    workers = []
    for bwType in bwTypes:
        for policy in policys:
            workers.append(Worker(modelDir, modelName, policy, traceIndex, cacheFile, bwType))

    [w.start() for w in workers]
    [w.join() for w in workers]
    '''
    for bwType in bwTypes:
        for policy in policys:
            worker = Worker(modelDir, modelName, policy, traceIndex, cacheFile, bwType)
            worker.run()



    '''
    for policy in policys:
        print("policy=", policy)
        worker = Worker(modelDir, modelName, policy, traceIndex, cacheFile, bwType)
        worker.run()
    '''
    # global PUREFLAF
    # PUREFLAF = True
    # if PUREFLAF == True:
    #     worker = Worker(modelDir, modelName, policys[1], traceIndex)
    # else:
    #     worker = Worker(modelDir, modelName, policys[0], traceIndex)
    # worker.run()


if __name__ == "__main__":
    main()