# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import platform



M_IN_K = 1000
BITRATES = [350, 600, 1000, 2000, 3000]
os.environ["OMP_NUM_THREADS"] = "1"
UPDATE_GLOBAL_ITER = 100
if platform.system() == "Linux":
    MAX_EP = 300000
    PRINTFLAG = False
else:
    MAX_EP = 40
    PRINTFLAG = True
GAMMA = 0.9
S_LEN = 5
A_DIM = 6
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
        prob = F.softmax(logits.view(1, -1), dim=1)
        m = self.distribution(prob)
        re = m.sample().numpy()[0]
        return np.int64(re), logits.data


    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values # advantage
        c_loss = td.pow(2) # value_loss = c_loss
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v # policy_loss = a_loss
        # entropy regularization ---
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(log_probs * probs).sum(1)
        a_loss -= 0.5 * entropy
        # entropy regularization ---
        total_loss = (c_loss + a_loss).mean()
        return total_loss


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name # worker name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net()        # local network
        # self.lnet.load_state_dict(torch.load("../../data/RL_model/2019-05-28_11-51-08/model/7318.pkl"))
        self.client = Client("train")
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]  # (video_name, popularity)
        self.busyTraceL = os.listdir("../../data/trace/busy/2")
        self.bwType = 3 # 默认用自生成带宽数据
        self.rttDict = {}
        get_rttDict(self.rttDict)


    def getBandwidthFile(self):
        self.bwType = random.randint(1,3)
        rtt = -1
        if self.bwType == 1:
            dir = "../../data/bandwidth/train/FCC"
        elif self.bwType == 2:
            dir = "../../data/bandwidth/train/HSDPA"
        else:
            dir = "../../data/bandwidth/train/mine"
        if self.bwType == 3:
            ipDirList = os.listdir(dir)
            ip = random.choice(ipDirList)
            if ip not in self.rttDict:
                print("no this ip:", ip, "in the bandwidth directory")
                dir = "../../data/bandwidth/train/HSDPA"
            else:
                rtt = self.rttDict[ip]
            bandwidthFileList = os.listdir(dir + "/" + ip)
            fileName = dir + "/" + ip + "/" + random.choice(bandwidthFileList)
        else:
            bandwidthFileList = os.listdir(dir)
            fileName = dir + "/" + random.choice(bandwidthFileList)
        # print(fileName)
        return fileName, rtt


    def get_busyTrace(self):
        fileName = self.busyTraceL[random.randint(0, len(self.busyTraceL) - 1)]
        return np.loadtxt("../../data/trace/busy/2/" + fileName).flatten().tolist()


    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            # get video -----------------------------
            while(True):
                video_random = random.random()
                videoName = ""
                for i in range(len(self.videoList)):
                    if video_random < self.videoList[i][1]:
                        videoName = self.videoList[i - 1][0]
                        break
                if videoName == "":
                    videoName = self.videoList[-1][0]
                else:
                    break

            # get video -----------------------------
            busyList = self.get_busyTrace()
            bandwidth_fileName, rtt = self.getBandwidthFile()
            reqBI = self.client.init(videoName, bandwidth_fileName, rtt, self.bwType)
            # mask---------------------------
            mask = [1] * A_DIM
            randmCachedBICount = random.randint(1, 5)
            BI = [0,1,2,3,4]
            randomCachedBI = random.sample(BI, randmCachedBICount)
            for bIndex in range(5):
                if bIndex not in randomCachedBI:
                    mask[bIndex] = 0
            # mask---------------------------
            segNum = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.

            busy = busyList[segNum%len(busyList)]
            state_ = np.zeros(S_LEN)
            state = state_.copy() #state =[reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, busy, mask]

            reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum, busy = [0] * 10
            # start one epoch **********************************
            while True:
                if sum(mask) == 1:
                    a = mask.index(1)
                    break
                # lnet.chose_action ****************************
                a, logits = self.lnet.choose_action(mask, v_wrap(state[None, :]))
                # lnet.choose_action ****************************
                # print --------------------------------------------
                if platform.system() == "Linux":
                    if random.randint(0,1000) == 1:
                        print("reqb=", reqBitrate, "lb=", lastBitrate, "buffer=", int(buffer), "hT=", int(hThroughput), "mT=", int(mThroughput), "busy=", round(busy, 2),
                              "mask=",mask, "action=", a, "reqBI=", reqBI, "reward=",round(reward,2), "logits=", logits)
                else:
                    print("reqb=", reqBitrate, "lb=", round(lastBitrate,2), "buffer=", int(buffer), "hT=", int(hThroughput), "mT=", int(mThroughput), "busy=", round(busy, 2),
                      "mask=",mask, "action=", a, "reqBI=", reqBI, "reward=",round(reward,2), "logits=", logits)
                # print --------------------------------------------
                busy = busyList[segNum % len(busyList)]
                # client.run ****************************
                if a == 5:
                    hitFlag = False
                else:
                    hitFlag = True
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = self.client.run(a, busy, hitFlag)
                # client.run ****************************
                state_[0] = reqBitrate / BITRATES[-1]
                state_[1] = lastBitrate / BITRATES[-1]
                state_[2] = (buffer/1000 - 30) / 10
                state_[3] = (hThroughput - throughput_mean) / throughput_std
                state_[4] = (mThroughput - throughput_mean) / throughput_std
                print(state)
                # state_[5] = (busy - busy_mean) / busy_std
                reward = reward / 5

                ep_r += reward
                buffer_a.append(a)
                buffer_s.append(state)
                buffer_r.append(reward)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, state_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                state = state_.copy()
                total_step += 1

        self.res_queue.put(None)
        print("end run")


def main():
    if PRINTFLAG == False:
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
        resFileDir = "../../data/RL_model/"+dt
        modelFileDir = resFileDir + "/model"
        if os.path.exists(modelFileDir) == False:
            os.makedirs(modelFileDir)

    gnet = Net()        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # parallel training
    print("cpu_count = %d" % mp.cpu_count())
    if platform.system() == "Linux":
        worker_count = 20
    else:
        worker_count = 1
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(worker_count)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    max_r = [-1,-1000]
    if PRINTFLAG == False:
        file = open(resFileDir+"/res.txt", 'w')
    epoch = 0
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            if r > max_r[1]:
                max_r[0] = epoch
                max_r[1] = r
                if PRINTFLAG==False:
                    torch.save(gnet.state_dict(), modelFileDir + "/" + str(epoch) + ".pkl")
                print("max r = %f, epoch = %d" % (max_r[0], max_r[1]))
            elif epoch % 100==0:
                if PRINTFLAG==False:
                    torch.save(gnet.state_dict(), modelFileDir + "/" + str(epoch) + ".pkl")
            if PRINTFLAG == False:
                file.write(str(r) + "\n")
                file.flush()
            epoch += 1
        else:
            break
    if PRINTFLAG == False:
        file.close()
    [w.join() for w in workers]
    if PRINTFLAG == False:
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(resFileDir+"/res.pdf")



if __name__ == "__main__":
    main()