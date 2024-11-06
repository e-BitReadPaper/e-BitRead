import sys
sys.path.append("..")
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
import matplotlib.colors as col
import seaborn as sns
import util

# input = size hThroughput mThroughput hTime mTime rtt maxMT reqCount totalMT avgMT

def cal_normal_parameter(data):
    mean = np.average(data)
    std = np.std(data)
    return mean, std

def get_zScore(dir="../../data/throughputRelation/data/", dataIndex = "0"):
    sample_data = np.loadtxt(dir+dataIndex+"/train/sample.csv", skiprows=1)
    eleList = ["size", "hThroughput", "mThroughput", "hTime", "mTime", "rtt", "maxMT", "reqCount", "totalMT", "avgMT"]

    meanList = []
    stdList = []
    for i in range(10):
        mean, std = cal_normal_parameter(sample_data[:,i])
        meanList.append(mean)
        stdList.append(std)

    print("meanList=",meanList)
    print("stdList=",stdList)
    print("eleList=",eleList)

def get_rttDict():
    lines = open("../file/rtt/rtt.txt").readlines()
    rttDict = {}

    for line in lines:
        ip = line.split(" ")[0]
        rtt = line.split(" ")[1].strip()
        rttDict[ip] = rtt
    return rttDict


def gen_sampleFile(dataIndex = "0", traceIndex = "0"):
    HISLEN=10
    traceDir = "../../data/trace/throughput_client/trace" + traceIndex
    resultDir = "../../data/throughputRelation/data/"
    trainDir = resultDir+str(dataIndex)+"/train"
    testDirt = resultDir+str(dataIndex)+"/test"
    if os.path.exists(trainDir) == False:
        os.makedirs(trainDir)
    if os.path.exists(testDirt) == False:
        os.makedirs(testDirt)

    fileTrain = open(trainDir + "/sample.csv", 'w')
    fileTest = open(testDirt + "/sample.csv", 'w')
    file = open(resultDir+str(dataIndex)+"/sample.csv",'w')
    firstRow = "size hThroughput mThroughput hTime mTime rtt maxMT reqCount totalMT avgMT\n"
    fileTrain.write(firstRow)
    fileTest.write(firstRow)
    file.write(firstRow)

    fileNameList = os.listdir(traceDir)
    fileCount = len(fileNameList)

    rttDict = get_rttDict()
    eleList = ["segNum", "size", "hThroughput", "mThroughput", "hTime", "mTime", "time", "rtt",
               "maxMT", "reqCount", "totalMT", "avgMT"]
    selectedEleList = ["size", "hThroughput", "mThroughput", "hTime", "mTime", "rtt", "maxMT", "reqCount", "totalMT", "avgMT"]
    selectedIndexList = []
    for i in selectedEleList:
        selectedIndexList.append(eleList.index(i))

    for i in range(fileCount):
        fileName = fileNameList[i]
        filePath = traceDir+"/"+fileName
        lines = open(filePath).readlines()[1:]

        hisEleList = [] # history selected element list
        for i in range(len(selectedEleList)):
            hisEleList.append([])

        randNum = random.random()

        for i in range(len(lines)):
            line = lines[i]
            elements = line.strip().split("\t")
            skipFlag = False
            newLine = ""
            for j in range(len(elements)):
                if j not in selectedIndexList:
                    continue
                elements[j] = float(elements[j])
                if eleList.index("rtt")==j and elements[j] == -1:
                    ip = fileName.split("_")[-1][0:-4]
                    if ip in rttDict:
                        elements[j] = float(rttDict[ip])
                    else:
                        print(ip)
                        skipFlag = True
                        break
                if eleList.index("mThroughput")==j and (elements[j]<0 or elements[j]>1e8):
                    skipFlag = True
                    break
                if eleList.index("hThroughput")==j and (elements[j]<0 or elements[j]>2e7):
                    skipFlag = True
                    break
                if eleList.index("maxMT")==j and (elements[j]<0 or elements[j]>1e8):
                    skipFlag = True
                    break

                eleName = eleList[j]
                selectedEleIndex = selectedEleList.index(eleName)
                if len(hisEleList[selectedEleIndex])<HISLEN: # line0 to lien9
                    hisEleList[selectedEleIndex].append(elements[j])
                    skipFlag = True
                    continue
                else:
                    hisEleList[selectedEleIndex] = hisEleList[selectedEleIndex][1:] + [elements[j]]
                elements[j] = str(sum(hisEleList[selectedEleIndex])/HISLEN)
                newLine = newLine+elements[j]
                if selectedEleIndex==len(selectedIndexList)-1:
                    newLine = newLine + "\n"
                else:
                    newLine = newLine + " "
            if skipFlag:
                continue

            if randNum < 0.8:
                fileTrain.write(newLine)
            else:
                fileTest.write(newLine)
            file.write(newLine)

    fileTest.close()
    fileTrain.close()
    file.close()


def regression(plotFlag = False):
    dataIndex = "1"
    data = pandas.read_csv("../../data/throughputRelation/data/"+dataIndex+"/sample.csv", delimiter=' ')
    if plotFlag:
        scatter_matrix(data[["size", "hThroughput", "mThroughput", "hTime", "mTime", "rtt", "maxMT", "reqCount", "totalMT", "avgMT"]],
                       figsize=(10, 10), diagonal='kde')
    plt.show()


def cleanRtt():
    file = open("./file/rtt_raw.txt")
    lines = file.readlines()
    print(lines)
    file_new = open("./file/rtt.txt",'w')
    i = 0
    while i < len(lines):
        if lines[i] == "":
            i += 1
            continue
        ip = ""
        if lines[i].split(" ")[-1].strip() == "bytes":
            print(lines[i])
            ip = lines[i].split(":")[0]
        else:
            i += 1
            continue
        while lines[i].split(" ")[0].strip() != "rtt":
            i += 1
        print(lines[i])
        rtt = (lines[i].split("=")[1]).split("/")[1]
        i += 1
        print("ip=%s rtt=%s" % (ip, rtt))
        file_new.write(ip+" "+rtt+"\n")


def draw_heatmap(HIT_FLAG=True):
    df = pd.read_csv('../../data/throughputRelation/data/3/sample.csv', delimiter=' ')
    # df.drop(['maxMT', 'avgMT', 'totalMT'], axis=1, inplace=True)
    # saleprice correlation matrix
    k = 7  # number of variables for heatmap
    corrmat = df.corr()
    if HIT_FLAG:
        cols = corrmat.nlargest(k, 'hThroughput')['hThroughput'].index
    else:
        cols = corrmat.nlargest(k, 'mThroughput')['mThroughput'].index
    cm = np.corrcoef(df[cols].values.T)

    fifthColor = '#7f0523'
    forthColor = '#df8064'
    thirdColor = '#ffffff'
    secondColor = '#70b0d1'
    firstColor = '#073a70'
    cmap = col.LinearSegmentedColormap.from_list('own2', [firstColor, secondColor, thirdColor, forthColor, fifthColor])

    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values, cmap=cmap, vmin=-1, vmax=1)
    dt = util.get_date(timeFlag=False)
    if HIT_FLAG:
        plt.savefig("../plot/heatmap_h2m_" + dt + ".pdf", bbox_inches = 'tight')
    else:
        plt.savefig("../plot/heatmap_m2h_" + dt + ".pdf", bbox_inches = 'tight')
    plt.show()


def main():
    # cleanRtt()
    # gen_sampleFile(dataIndex = "0", traceIndex="0")
    get_zScore(dataIndex = "2")
    # regression(Tr    # draw_heatmap(HIT_FLAG=True)
    #     # draw_heatmap(HIT_FLAG=False)ue)



main()

