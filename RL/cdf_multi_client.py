import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm  # recommended import according to the docs
import math
from prettytable import PrettyTable

def drawCDF(arr, label):
    ecdf = sm.distributions.ECDF(arr)
    x = np.linspace(min(arr), max(arr))
    y = ecdf(x)
    plt.plot(x, y, linestyle='-', linewidth=1.5, label=label)


def cdf(folder):
    bw_types = ['mine', 'FCC','HSDPA']
    table = PrettyTable(["bw_type", "pol", "avg qoe", "avg rebuf",
                         "hit ratio", "avg bitrate variation", "avg bitrate"])
    for bw_type in bw_types:
        dir = folder + "/" + bw_type
        qoeL = [[], [], [], [], [], []]
        bitrateL = [[], [], [], [], [], []]
        for i in range(len(bitrateL)):
            bitrateL[i] = [0] * 5
        rebL = [[], [], [], [], [], []]
        bVarL = [[], [], [], [], [], []]
        hitRL = [0]*6
        # polDir = os.listdir(dir+"/trace"+str(cachePol))
        polDir = os.listdir(dir)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            segCount = 0
            total_rew = 0
            total_reb = 0.0
            total_osc = 0.0
            total_b = 0.0
            
            # 遍历所有客户端
            for client_id in range(10):  # 遍历所有客户端
                client_path = dir + "/" + pol + f"/client_{client_id}"
                if not os.path.exists(client_path):
                    continue
                    
                for traceFName in os.listdir(client_path):
                    file_path = os.path.join(client_path, traceFName)
                    try:
                        with open(file_path) as file:
                            lines = file.readlines()
                            # 跳过只有表头的文件（表头通常是第一行）
                            if len(lines) <= 1 or (len(lines) == 2 and not lines[1].strip()):
                                continue
                                
                            content_lines = [line for line in lines[1:] if line.strip()]  # 跳过第一行表头，并过滤空行
                            if not content_lines:  # 如果没有实际内容，跳过该文件
                                continue
                                
                            segCount += len(content_lines)
                            last_br = 0
                            
                            for line in content_lines:
                                elements = line.split("\t")
                                # hit 统计
                                hit = int(elements[2])
                                if hit == 1:
                                    hitCount += 1
                                    
                                # bitrate index 统计
                                bI = int(elements[5])
                                if bI != -1:
                                    bitrateL[polI][bI] += 1
                                    
                                # bitrate variation 统计
                                bitrate = int(elements[7])*1.0 / 1024  # Mbps
                                total_b += bitrate
                                total_osc += abs(last_br - bitrate)
                                bVarL[polI].append(abs(last_br - bitrate))
                                last_br = bitrate
                                
                                # rebuffer 统计
                                reb = max(0, float(elements[12]))
                                total_reb += reb
                                if reb > 20000:
                                    print(f"Client {client_id}:", traceFName, line)
                                rebL[polI].append(reb)
                                
                                # qoe 统计
                                qoe = float(elements[-3])
                                total_rew += qoe
                                qoeL[polI].append(qoe)
                            
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
                    
                    file.close()
            
            # 计算平均值
            if segCount > 0:  # 防止除零错误
                hitRL[polI] = 1.0 * hitCount/segCount
                table.add_row([
                    bw_type, 
                    pol, 
                    total_rew/segCount, 
                    total_reb/segCount, 
                    1.0 * hitCount/segCount, 
                    total_osc/segCount,
                    total_b/segCount
                ])
        
        # 绘图部分保持不变
        fig = plt.figure(figsize=(20,4))
        fontSize = 10
        
        # qoe-----------------------
        ax1 = fig.add_subplot(1,5,1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            drawCDF(np.array(qoeL[polI]), pol)
        plt.xlabel("qoe", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        # plt.xlim(0.8, 2.1)
        # plt.ylim(0.5, 1.01)
        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        ax2 = fig.add_subplot(1,5,2)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.15
        colors = ["#98133a", "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9", "#f15a22"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]

            rects = plt.bar(index + bar_width*(polI - 2.5), 1.0*bitrateL[polI, :]/np.sum(bitrateL[polI, :])*100, bar_width,
                            alpha=opacity, color=colors[polI],
                            label=pol)
        plt.xlabel("bitrate(kbps)", fontsize=fontSize)
        plt.ylabel("request count(%)", fontsize=fontSize)
        # plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        # rebuffer-----------------------
        ax3 = fig.add_subplot(1,5,3)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            drawCDF(np.array(rebL[polI]), pol)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("rebuffer time", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        # bitrate variation-----------------
        ax3 = fig.add_subplot(1,5,4)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            drawCDF(np.array(bVarL[polI]), pol)
        plt.legend(loc='best', fontsize=fontSize)
        plt.xticks(fontsize=fontSize)
        plt.yticks(fontsize=fontSize)
        plt.xlabel("bitrate variation", fontsize=fontSize)
        plt.ylabel("cdf", fontsize=fontSize)
        # hit ratio----------------
        ax3 = fig.add_subplot(1,5,5)
        n_groups = 6
        opacity = 0.8
        bar_width = 0.35
        index = np.arange(n_groups)
        plt.bar(index, np.array(hitRL)*100, bar_width)
        plt.ylabel('hit ratio(%)')
        plt.xticks(index, (polDir[0], polDir[1], polDir[2], polDir[3], polDir[4], polDir[5]))
        # plt.xticks(index, (polDir[0], polDir[1]))


        plt.savefig(dir + "/"  + bw_type + ".pdf")
        plt.show()

    print(table.get_string())


def takeSecond(elem):
    return elem[1]


def getSizeDict(dir):
    file = open("../file/videoSizePerVision.txt")
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

CACHERULE = 0

def makeRLCachedVirsion():
    file = open("./trace/logits.txt")
    logitsD = {}
    for line in file.readlines():
        virsionName = line.split(" ")[0]
        logits = float(line.split(" ")[1])
        if virsionName not in logitsD:
            logitsD[virsionName] = logits
        else:
            logitsD[virsionName] += logits

    virsionSizeDict = getSizeDict(dir)

    logitsL = []
    for virsionName in logitsD:
        logit = logitsD[virsionName]
        size = virsionSizeDict[virsionName]
        costBenifit = logit / size * 1000000
        logitsL.append([virsionName,costBenifit, logit, size])
    logitsL.sort(key=takeSecond, reverse=True)

    file = open("./trace/cachedVirsion_" + str(4) + ".txt", 'w')
    cachedSize = 0
    i = 0
    cachedVirsionList = []
    while cachedSize < 10 * 1024 * 1024 * 1024:  # 10GB
        if CACHERULE == 0:  # 不限制每个视频能存几个版本
            cachedSize += logitsL[i][3]
            cachedVirsionList.append(logitsL[i][0])
        else:
            existedVirsionCount = 0
            for j in range(1, 6):
                virsion = logitsL[i][0].split("_")[0] + "_" + str(j)
                if virsion in cachedVirsionList:
                    existedVirsionCount += 1
            if existedVirsionCount < CACHERULE:
                cachedSize += logitsL[i][3]
                cachedVirsionList.append(logitsL[i][0])
        i += 1

    cachedVirsionList.sort()
    for virsion in cachedVirsionList:
        file.write(virsion + "\n")

    file.close()


def main():
    dir = "./res/2019-03-19_19-36-28/trace_1"
    bw_types = ['mine', 'FCC',]

    dir = "../data/RL_model/2024-12-19_14-30-50/test_trace_multi_client_2/"
    cdf(dir)

    # drawClient()

    # makeRLCachedVirsion()

main()

