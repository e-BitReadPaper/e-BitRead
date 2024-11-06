# -*- coding: utf-8 -*
import os
import sys
sys.path.append("..")
import util
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable
import random

def cdf_segSize():
    labels = ["350", "600", "1000", "2000", "3000"]
    for bIndex in range(5):
        data = np.loadtxt("./file/segSize/segSize"+str(bIndex+1)+".txt")
        util.drawCDF(data=data/1024/1024, label=labels[bIndex], color=util.colorL[bIndex], pointNum=50)
    plt.xlabel("video chunk size(MB)", fontsize=25)
    plt.ylabel("CDF", fontsize=25)
    plt.legend(loc='best', fontsize=21)
    plt.tick_params(labelsize=23)
    plt.xlim(0, 2)
    plt.grid(ls='--')
    plt.savefig("./plot/segSize.pdf", bbox_inches = 'tight')
    plt.show()


def get_segSize():
    videoList = util.get_videoList(dir="./video.txt")
    videoDir = "/opt/nginx/html/video"
    for videoName in videoList:
        for bIndex in range(1,6):
            file = open("./segSize"+str(bIndex)+".txt", 'a')
            videoBitratePath = videoDir+"/"+videoName+"/video/avc1/"+str(bIndex)
            segTotalSizeThisBitrate = 0
            for segFileName in os.listdir(videoBitratePath):
                segSize = os.path.getsize(videoBitratePath+"/"+segFileName) # Byte
                file.write(str(segSize)+"\n")
            file.close()


def get_clientBitrate():
    bSet = [350, 600, 1000, 2000, 3000]
    rttDict = util.get_rttDict()
    traceDir = "../data/trace/pure/trace1"
    avgBList = []
    for fileName in os.listdir(traceDir):
        ip = fileName[:-4].split("_")[-1]
        if ip in rttDict:
            print(ip,"\t", rttDict[ip])
        else:
            print(ip, "not have rtt")
        filePath = traceDir + "/" + fileName
        lines = open(filePath).readlines()[1:]
        totalB = 0
        reqCount = 0
        for line in lines:
            elements = line.split("\t")
            bitrate = int(elements[6])/1024/1024
            bI = int(elements[4])
            totalB += bSet[bI]
            reqCount += 1

        avgBList.append(totalB*1.0/reqCount)

    util.drawCDF(np.array(avgBList), marker=util.markerL[0], pointNum=10, color="#000000")
    # plt.title("client average bitrate CDF")
    plt.xlabel("bitrate/kbps", fontsize=25)
    plt.ylabel("CDF", fontsize=25)
    plt.tick_params(labelsize=23)
    plt.grid(ls='--')
    plt.savefig("./plot/clientBitrate.pdf", bbox_inches = 'tight')
    plt.show()

    print("total bitrate=",sum(avgBList))
    print("miss total bitrate=", sum(avgBList)*0.6)

def get_clientRTT():
    rttDict = util.get_rttDict()
    valueL = []
    for ip in rttDict:
        valueL.append(float(rttDict[ip]))
    util.drawCDF(data=np.array(valueL), marker=util.markerL[0], color="#000000")
    # plt.title("client rtt CDF")
    plt.tick_params(labelsize=23)
    plt.xlabel("rtt/ms", fontsize=25)
    plt.ylabel("CDF", fontsize=25)
    plt.xticks([0, 100, 200, 300, 400])
    plt.grid(ls='--')
    plt.savefig("./plot/clientRTT.pdf", bbox_inches = 'tight')
    plt.show()

    print("average rtt=", sum(valueL)/len(valueL))

def compare_clientIP():
    ipList = open("./file/activeNode0411.txt").readlines()
    ipList = [i.strip() for i in ipList]
    traceIpL = []
    for fileName in os.listdir("../data/trace/throughput/trace3"):
        ip = fileName[:-4].split("_")[-1]
        print(ip)
        if ip not in traceIpL:
            traceIpL.append(ip)

    for ip in ipList:
        if ip not in traceIpL:
            print("no ", ip)

def get_busy():
    dir = "../data/trace/throughput_client/trace2"
    busyDir = "../data/trace/busy/2"
    if os.path.exists(busyDir) == False:
        os.makedirs(busyDir)

    for fileName in os.listdir(dir):
        filePath = dir + "/" + fileName
        ip = fileName[:-4].split("_")[-1]
        dt = fileName[:-4].split("_")[0] + "_" + fileName[:-4].split("_")[1]
        print(ip, dt)
        data = np.loadtxt(filePath, skiprows=1)
        busy = data[:, -3]
        np.savetxt(busyDir+"/"+ip+"_"+dt+".txt", busy)

def takeSecond(elem):
    return elem[1]

def getVirsionFrequencyDict(traceDir):
    dir = traceDir
    urlDict = {}
    for traceFileName in os.listdir(dir):
        videoName = traceFileName.split("_")[2]
        traceFile = open(dir + "/" + traceFileName)
        lines = traceFile.readlines()[1:]
        for line in lines:
            elements = line.split("\t")
            # No	cSize	Hit	buffer	bI	aBI	lastBI	bitrate	throughput	hThroughput	mThroughput	downloadT	rebufferT	qoe	reward	time	busy
            size = float(elements[1].strip())
            bIndex = int(elements[4].strip()) + 1 # 1-5
            url = videoName+"_"+str(bIndex)
            if url not in urlDict:
                urlDict[url] = 1
            else:
                urlDict[url] += 1

    return urlDict


def getSizeDict():
    file = open("./file/videoSizePerVision.txt")
    lines = file.readlines()
    virsionSizeDict = {} # videoName_bIndex size
    for line in lines:
        virsion = line.split(" ")[0]
        size = int(line.split(" ")[1])
        if virsion not in virsionSizeDict:
            virsionSizeDict[virsion] = size
        else:
            print(virsion,"show twice")
    return virsionSizeDict


def getTotalSize():
    file = open("../makeVideoSet/videoSizePerVision.txt")
    lines = file.readlines()
    totalSize = 0
    for line in lines:
        virsion = line.split(" ")[0]
        size = int(line.split(" ")[1])
        totalSize += size
    print("total virsion size= %d GB" % (totalSize*1.0/1024/1024/1024))


def getCountDict():
    file = open("./file/chunkCount.txt")
    lines = file.readlines()
    countDict = {} # videoName_bIndex size
    for line in lines:
        videoN = line.split(" ")[0]
        count = int(line.split(" ")[1])
        if videoN not in countDict:
            countDict[videoN] = count
        else:
            print(videoN,"show twice")
    return countDict


def get_cachedVideo(traceDir, outputLabel):
    resultDir = "./file"
    virsionSizeDict = getSizeDict()
    virsionFreqDict = getVirsionFrequencyDict(traceDir=traceDir)
    chunkCountDict = getCountDict()
    virsionList = []
    for key in virsionFreqDict:
        frequency = virsionFreqDict[key]
        size = virsionSizeDict[key]
        costBenifit = frequency / size * 1000000
        virsionList.append([key, costBenifit, frequency, size])
    virsionList.sort(key=takeSecond, reverse=True)
    # file = open("./virsionBenifit.txt",'w')
    # for line in virsionList:
    #     file.write(line[0]+" "+str(line[1])+" "+str(line[2])+" "+str(line[3])+"\n")
    # file.close()
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y%m%d", time_local)

    file = open(resultDir + "/cachedFile_" + dt + "_" + outputLabel + ".txt", 'w')
    cachedSize = 0
    i = 0
    cachedVirsionList = []
    while cachedSize < 10 * 1024 * 1024 * 1024:  # 10GB
        cachedSize += virsionList[i][3]
        cachedVirsionList.append(virsionList[i][0].split("_")[0] + " " + virsionList[i][0].split("_")[1])
        i += 1

    for virsion in cachedVirsionList:
        file.write(virsion + " " + str(chunkCountDict[virsion.split(" ")[0]]) + "\n")
    file.close()


def cdf_bandwidth():
    # matplotlib.use('agg')
    dir = "../data/bandwidth/train/"
    bwTypes = ["FCC", "HSDPA", "mine"]
    # fig = plt.figure(figsize=(12,5))
    for bwType in bwTypes:
        bwList = []
        path = dir + bwType

        if bwType.find('mine') != -1:
            print()
            ipList = os.listdir(path)
            for ip in ipList:
                path_ = path + "/" + ip
                for fileName in os.listdir(path_):
                    data = np.loadtxt(path_ + "/" + fileName)[:, 1]
                    dataL = data.flatten().tolist()
                    for d in range(len(dataL)):
                        if dataL[d] > 11*1024*1024:
                            dataL[d] = 11*1024*1024
                    bwList = bwList + dataL

        else:
            for fileName in os.listdir(path):
                data = np.loadtxt(path + "/" + fileName)[:, 1]
                dataL = data.flatten().tolist()
                for d in range(len(dataL)):
                    if dataL[d] > 11 * 1024 * 1024:
                        dataL[d] = 11 * 1024 * 1024
                bwList = bwList + dataL
        bwIndex = bwTypes.index(bwType)
        util.drawCDF(data=np.array(bwList)/1024/1024,
                     pointNum=10,
                     label=bwType, color=util.colorL[bwIndex], marker=util.markerL[bwIndex])

    # plt.title("defferent type bandwidth compare")
    plt.tick_params(labelsize=23)
    plt.xlabel("bandwidth(Mbps)", fontsize=25)
    plt.ylabel("CDF", fontsize=25)
    plt.grid(ls='--')
    plt.legend(loc='best', fontsize=23)
    plt.xlim(0, 10)

    plt.savefig("./plot/bandwidth.pdf", bbox_inches = 'tight')
    plt.show()

def cdf_policy(dir):
    table = PrettyTable(["bwType", "policy", "avg QoE", "avg reward", "avg rebuffer", "hit ratio", "byte hit ratio", "avg bitrate variation", "avg bitrate"])
    bwTypes = ["FCC", "HSDPA", "mine"]
    for bwType in bwTypes:
        qoeL = []
        bitrateL = []
        rebL = [[]]
        bVarL = [[]]
        hitRL = []
        polDir = os.listdir(dir + bwType)
        polCount = len(polDir)
        for i in range(polCount):
            qoeL.append([])
            bitrateL.append([0] * 5)
            rebL.append([])
            bVarL.append([])
            hitRL.append(0)

        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            hitByte = 0
            segCount = 0
            total_rew = 0
            total_qoe = 0.
            total_reb = 0.0
            total_osc = 0.0
            total_b = 0.0
            for traceFName in os.listdir(dir + bwType + "/" + pol):
                file = open(dir + bwType + "/" + pol + "/" + traceFName)
                lines = file.readlines()[1:]
                segCount += len(lines)
                last_br = 0
                for line in lines:
                    # No cSize 2Hit buffer bI aBI lastBI 7bitrate throughput hThroughput mThroughput downloadT
                    # 12rebufferT qoe reward time busy
                    elements = line.split("\t")
                    # bitrate index ---------------------------
                    bI = int(elements[5])
                    bitrateL[polI][bI] += 1
                    # bitrate variation -----------------------
                    bitrate = int(elements[7]) * 1.0 / 1024 / 1024  # Mbps
                    total_b += bitrate
                    total_osc += abs(last_br - bitrate)
                    bVarL[polI].append(abs(last_br - bitrate))
                    last_br = bitrate
                    # hit -------------------------------------
                    hit = int(elements[2])
                    if hit == 1:
                        hitCount += 1
                        hitByte += bitrate
                    # rebufferT -------------------------------
                    reb = float(elements[-5])
                    total_reb += reb
                    if reb > 20000:
                        print(traceFName, line)
                    # if reb > 5:
                    #     reb = 5
                    rebL[polI].append(reb)
                    # qoe -------------------------------------
                    qoe = float(elements[-4])
                    # if qoe < -5:
                    #     qoe = -5
                    total_qoe += qoe
                    qoeL[polI].append(qoe)
                    # reward -----------------------------------
                    reward = float(elements[-3])
                    total_rew += reward

            hitRL[polI] = 1.0 * hitByte / total_b
            # print(bwType, pol, "avg qoe=", total_rew / segCount, ",avg rebuf=", total_reb / segCount, ",hit ratio=",
            #       1.0 * hitCount / segCount, ",byte hit ratio=", 1.0 * hitByte / total_b, ",avg bitrate variation=",
            #       total_osc / segCount, ",avg bitrate=", total_b / segCount)
            table.add_row([bwType, pol, total_qoe / segCount, total_rew / segCount, total_reb / segCount,
                  1.0 * hitCount / segCount, 1.0 * hitByte / total_b,
                  total_osc / segCount,  total_b / segCount])


        fig = plt.figure(figsize=(20, 4))
        fontSize = 10
        # qoe-----------------------
        ax1 = fig.add_subplot(1, 5, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            util.drawCDF(np.array(qoeL[polI]), pol, color=util.colorL[polI])
        plt.xlabel("qoe", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        # plt.xlim(0.8, 2.1)
        # plt.ylim(0.5, 1.01)
        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        ax2 = fig.add_subplot(1, 5, 2)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.15
        colors = ["#98133a", "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9", "#A9CADB", "#d55e00"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]

            rects = plt.bar(index + bar_width * (polI - 2.5), 1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100,
                            bar_width,
                            alpha=opacity, color=colors[polI],
                            label=pol)
        plt.xlabel("bitrate(kbps)", fontsize=fontSize)
        plt.ylabel("request count(%)", fontsize=fontSize)
        # plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        # rebuffer-----------------------
        ax3 = fig.add_subplot(1, 5, 3)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            util.drawCDF(np.array(rebL[polI]), pol, color=util.colorL[polI])
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("rebuffer time", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        # bitrate variation-----------------
        ax3 = fig.add_subplot(1, 5, 4)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            util.drawCDF(np.array(bVarL[polI]), pol, color=util.colorL[polI])
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("bitrate variation", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        # hit ratio----------------
        ax3 = fig.add_subplot(1, 5, 5)
        n_groups = polCount
        opacity = 0.8
        bar_width = 0.35
        index = np.arange(polCount)
        plt.bar(index, np.array(hitRL), bar_width)
        plt.ylabel('hit ratio(%)')
        ticksList = []
        for i in range(polCount):
            ticksList.append(polDir[i])
        # plt.xticks(index, (polDir[0], polDir[1], polDir[2], polDir[3], polDir[4]))
        plt.xticks(index, ticksList)
        plt.title(bwType)
        plt.savefig("./plot/cdf_policy_hitratio.pdf")
        plt.show()

    print(table.get_string())

def cdf_hitMissThroughput():
    data = np.loadtxt("../data/throughputRelation/data/2/train/sample.csv", skiprows=1)
    # size hThroughput mThroughput hTime mTime rtt maxMT reqCount totalMT avgMT
    data_hThroughput = data[:, 1].flatten()
    data_mThroughput = data[:, 2].flatten()
    print(np.average(data_hThroughput))
    print(np.average(data_mThroughput))
    # fig = plt.figure(figsize=(4, 3))
    util.drawCDF(data=data_hThroughput/1024/1024, label="hit throughput", marker=util.markerL[0], color=util.colorL[0])
    util.drawCDF(data=data_mThroughput/1024/1024, label="miss throughput", marker=util.markerL[1], color=util.colorL[1])
    plt.legend(loc='best', fontsize=20)
    plt.tick_params(labelsize=23)
    plt.xlabel("Throughput(Mbps)", fontsize=25)
    plt.ylabel("CDF", fontsize=25)
    dt = util.get_date(timeFlag=False)
    plt.grid(ls='--')
    plt.savefig("./plot/cdf_hitMissThroughput_"+dt+".pdf", bbox_inches = 'tight')
    plt.show()

def get_AverageSessionDuration():
    dir = "../data/trace/throughput/trace3"
    sum_timeDuration = 0
    counter = 0
    for fileName in os.listdir(dir):
        filePath = dir + "/" + fileName
        data = np.loadtxt(filePath, skiprows=1)
        time_start = data[0, 6]
        time_end = data[-1, 6]
        timeDuration = time_end - time_start
        print(time_start, time_end, timeDuration)
        sum_timeDuration += timeDuration
        counter += 1
    print("avg_timeDuration=", sum_timeDuration / counter, "秒")


def getBandwidthCount(bwType):
    dir = "../data/bandwidth/test/" + bwType
    bwCount = 0
    if bwType == "mine":
        ipDirList = os.listdir(dir)
        for ip in ipDirList:
            if os.path.exists(dir + "/" + ip) == False:
                continue
            bandwidthFileList = os.listdir(dir + "/" + ip)
            bwCount += len(bandwidthFileList)

    else:
        bandwidthFileList = os.listdir(dir)
        bwCount += len(bandwidthFileList)
    return bwCount


def getVideoTrace():
    bwTypes = ["FCC", "HSDPA", "mine"]
    video_file = open("./file/videoList/video.txt")
    videoList = video_file.readlines()
    videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in videoList]
    bwCount_max = -1
    for bwType in bwTypes:
        bwCount = getBandwidthCount(bwType)
        if bwCount > bwCount_max:
            bwCount_max = bwCount

    videoTraceL = []
    videoTraceFile = open("./file/videoTrace.txt", 'w')
    while len(videoTraceL) < bwCount:
        videoName = ""
        video_random = random.random()
        for i in range(len(videoList)):
            if video_random < videoList[i][1]:
                videoName = videoList[i - 1][0]
                break
        if videoName == "":
            videoName = videoList[-1][0]
        videoTraceL.append(videoName)
        videoTraceFile.write(videoName + "\n")
    videoTraceFile.close()


def get_videoPopularity():
    video_count = 50
    file = open("./file/videoList/video.txt")
    video_name_list = file.readlines()
    video_name_list = [i.strip().split(" ")[0] for i in video_name_list]
    print(video_name_list)
    random.shuffle(video_name_list)
    print("shuffle", video_name_list)
    file.close()
    x = range(1, video_count + 1)
    c = -0.73
    y = [i ** c for i in x]
    print(y)
    s = sum(y)
    y_prob = [i / s for i in y]
    print(y_prob)
    plt.xlabel("Rank of video files", fontsize=25)
    plt.ylabel("Request probability", fontsize=25)
    plt.tick_params(labelsize=23)
    plt.grid(ls='--')
    plt.plot(x, y, linestyle='-', linewidth=4, marker=util.markerL[1], color="#000000",
             markeredgecolor="#000000", markerfacecolor="#000000", markersize=10, markeredgewidth=3)
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.savefig("./plot/popularity.pdf", bbox_inches = 'tight')
    plt.show()

    # y_prob_sum = []
    # s = 0
    # for i in range(video_count):
    #     y_prob_sum.append(s + y_prob[i])
    #     s += y_prob[i]
    # print(y_prob_sum)
    #
    # video_popularity = [(video_name_list[i], y_prob_sum[i]) for i in range(video_count)]
    # print(video_popularity)
    #
    # file = open("./file/video_new.txt", "w")
    # for popularity in video_popularity:
    #     video_type = random.randint(1, 5)
    #     file.write(popularity[0] + " " + str(popularity[1]) + "\n")
    # file.close()


def draw_trace_time_element(element='reqCount'):
    dir = "../data/trace/throughput_client/trace2"
    elementDict = {} # {timestamp:[totalreqCount, count]}
    minTime = 1e9
    elements = ["segNum","size","hThroughput","mThtoughput","hTime","mTime","time","rtt","maxMT","reqCount","totalMT","avgMT"]
    for fileName in os.listdir(dir):
        lines = open(dir + "/" + fileName).readlines()[1:]
        for line in lines:
            parses = line.split("\t")
            time = int(int(parses[6])/10)
            if time < minTime:
                minTime = time
            element_value = float(parses[elements.index(element)].strip())
            if time in elementDict:
                elementDict[time][0] += element_value
                elementDict[time][1] += 1
            else:
                elementDict[time] = [0, 0]
                elementDict[time][0] = element_value
                elementDict[time][1] = 1
    elementList = []
    for time in elementDict:
        elementList.append([time, elementDict[time][0] / elementDict[time][1]])
    elementList.sort(reverse=True)

    data = np.array(elementList)
    plt.plot((data[:, 0] - minTime)*10, data[:, 1])
    plt.xlabel("time(s)")
    plt.ylabel(element)
    plt.show()


def get_minRTTandMaxRTT():
    lines = open("./file/rtt/rtt.txt").readlines()
    lines = [float(i.strip().split(" ")[1]) for i in lines]
    print(max(lines), min(lines))


def get_videoDuration():
    max = 0
    min = 1000000
    videoL = open("./file/videoList/video.txt").readlines()
    videoL = [i.strip().split(" ")[0] for i in videoL]
    print(videoL)
    for fileName in os.listdir("./file/videoSize/"):
        print(fileName)
        lines = open("./file/videoSize/" + fileName).readlines()
        if fileName.split(".")[0].strip() in videoL:
            if max < len(lines):
                max = len(lines)
            if min > len(lines):
                min = len(lines)
                print(fileName)
    print(max*4/60, min*4/60)


def get_bandwidth_avg_std():
    dir = "../data/bandwidth/train/"
    data = np.zeros(1)
    for bwType in os.listdir(dir):
        if bwType == "mine-miss":
            continue
        if bwType == "mine":
            for ip in os.listdir(dir + bwType):
                for fileName in os.listdir(dir + bwType + "/" + ip):
                    data_ = np.loadtxt(dir + bwType + "/" + ip + "/" +fileName)[:, 1].flatten()
                    if data.size == 1:
                        data = data_
                    else:
                        data = np.concatenate((data, data_), axis=0)

        else:
            for fileName in os.listdir(dir + bwType):
                data_ =  np.loadtxt(dir + bwType + "/" + fileName)[:, 1].flatten()
                if data.size == 1:
                    data = data_
                else:
                    data = np.concatenate((data, data_), axis=0)

    print("size=", data.size)
    avg = np.average(data)
    std = np.std(data)
    print("avg=", avg)
    print("std=", std)
    '''
    size= 1462284
    avg= 2297514.2311790097
    std= 4369117.906444455
    '''


def main():
    # get_segSize()
    # cdf_segSize()

    # get_clientBitrate()

    # get_clientRTT()

    # compare_clientIP()

    # get_busy()

    '''
    traceDirList = ["../data/RL_model/2019-05-28_11-51-08/trace_1/no_policy/2019-06-01_18-44-35/mine",
                    "../data/RL_model/2019-05-28_11-51-08/trace_1/no_policy/2019-06-01_18-40-43/HSDPA",
                    "../data/RL_model/2019-05-28_11-51-08/trace_1/no_policy/2019-06-01_18-39-18/FCC"]
    labels = ["mine", "HSDPA", "FCC"]

    for i in range(3):
        get_cachedVideo(traceDir= traceDirList[i], outputLabel=labels[i])
    '''
    # traceDir = "../data/trace/pure/trace1"
    # get_cachedVideo(traceDir=traceDir, outputLabel="pure")

    ''''''
    cdf_bandwidth()


    # cdf_policy(dir="../data/RL_model/2019-06-01_16-33-11/trace_1/")

    # cdf_hitMissThroughput()

    # get_AverageSessionDuration()

    # getVideoTrace()

    # get_videoPopularity()

    # draw_trace_time_element(element="reqCount")

    # get_minRTTandMaxRTT()

    # get_videoDuration()

    # get_bandwidth_avg_std()

main()