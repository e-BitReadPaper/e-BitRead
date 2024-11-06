import os
import numpy as np
import random
import util

def get_activeNodes():
    ipList = open("./file/activeNode0411.txt").readlines()
    ipList = [i.strip() for i in ipList]
    return ipList


def clean_rtt(fileDir, fileName):
    file = open(fileDir + "/" + fileName)
    lines = file.readlines()
    file_new = open(fileDir + "/rtt.txt", 'w')
    activeNodes = get_activeNodes()
    i = 0
    ipRttDict = {}
    while i < len(lines):
        if lines[i] == "":
            i += 1
            continue
        ip = ""
        if lines[i].find("bytes")!=-1 and lines[i].find("0 bytes")==-1:
            ip = lines[i].split(":")[0]
        else:
            i += 1
            continue
        while lines[i].split(" ")[0].strip() != "rtt":
            i += 1
        rtt = (lines[i].split("=")[1]).split("/")[1]
        i += 1
        if ip not in ipRttDict:
            ipRttDict[ip] = rtt
        else:
            print(ip, "show up twice")
        print("ip=%s rtt=%s" % (ip, rtt))
        file_new.write(ip + " " + rtt + "\n")

    for ip in activeNodes:
        if ip not in ipRttDict:
            print(ip,"is active but no rtt")


def cook_fccBwTrace():
    FILE_PATH = '../../../data/FCC/data-raw-2018-jan/201801/curr_webget.csv'
    NUM_LINES = np.inf
    line_counter = 0
    bw_measurements = {}

    with open(FILE_PATH) as f:
        firstLine = True
        for line in f:
            if firstLine == True:
                firstLine = False
                continue
            parse = line.split(',')

            uid = parse[0]
            target = parse[2]
            address = parse[3]
            throughput = int(parse[6])*8  # bit per second

            k = (uid, target)
            if k in bw_measurements:
                bw_measurements[k].append(throughput)
            else:
                bw_measurements[k] = [throughput]

            line_counter += 1
            if line_counter >= NUM_LINES:
                break
    lines = []
    lineCounter = 0
    for k in bw_measurements:
        lines = lines + bw_measurements[k]
        lineCounter += len(bw_measurements[k])
        if lineCounter > 1000*1200:
            break
    print(lineCounter)
    OUTPUT_PATH_train = '../data/bandwidth/train/FCC/'
    OUTPUT_PATH_test = '../data/bandwidth/test/FCC/'
    if os.path.exists(OUTPUT_PATH_train) == False:
        os.makedirs(OUTPUT_PATH_train)
    if os.path.exists(OUTPUT_PATH_test) == False:
        os.makedirs(OUTPUT_PATH_test)
    lineCounter = 0
    traceCounter = 0
    testCounter = 0
    lastTimestamp = -1
    while traceCounter < 1200:
        if random.random() < 0.2 and testCounter < 200:
            OUTPUT_PATH = OUTPUT_PATH_test
            testCounter += 1
        else:
            OUTPUT_PATH = OUTPUT_PATH_train
        outputFile = open(OUTPUT_PATH + str(traceCounter)+".log", 'w')
        time = 0
        while(True):
            if lineCounter>=len(lines):
                lineCounter = 0
            throughput = lines[lineCounter]
            outputFile.write(str(time) + " " + str(throughput) + "\n")
            lineCounter += 1
            time+=5
            if time>1000:
                time = 0
                break
        traceCounter += 1


def cook_hsdpaBwTrace():
    dir = "../../../data/HSDPA/bandwidthData/bandwidthData"
    lines = []
    for fileName in os.listdir(dir):
        lines = lines + open(dir+"/"+fileName).readlines()
    OUTPUT_PATH_train =  '../data/bandwidth/train/HSDPA/'
    OUTPUT_PATH_test = '../data/bandwidth/test/HSDPA/'
    if os.path.exists(OUTPUT_PATH_train) == False:
        os.makedirs(OUTPUT_PATH_train)
    if os.path.exists(OUTPUT_PATH_test) == False:
        os.makedirs(OUTPUT_PATH_test)
    print("line Count=", len(lines))
    lineCounter = 0
    traceCounter = 0
    testCounter = 0
    while traceCounter < 1200:
        if random.random() < 0.2 and testCounter < 200:
            OUTPUT_PATH = OUTPUT_PATH_test
            testCounter += 1
        else:
            OUTPUT_PATH = OUTPUT_PATH_train
        outputFile = open(OUTPUT_PATH + str(traceCounter)+".log", 'w')
        for i in range(1000):
            if lineCounter>=len(lines):
                lineCounter = 0
            line = lines[lineCounter]
            outputFile.write(str(i) + " " + line.split(" ")[1])
            lineCounter += 1
        traceCounter += 1


def cook_myBwTrace(hitFlag=True):
    linesD = {}
    dir = "../data/trace/throughput_client/trace2"
    for fileName in os.listdir(dir):
        newLines = open(dir + "/" + fileName).readlines()[1:]
        ip = fileName[:-4].split("_")[-1] # 2019-05-13_12-56-57_short18_204.8.155.226.csv
        if ip not in linesD:
            linesD[ip] = newLines
        else:
            linesD[ip] = linesD[ip] + newLines
    # for ip in linesD:
    #     print(ip)
    if hitFlag:
        OUTPUT_PATH_train =  '../data/bandwidth/train/mine-hit/'
        OUTPUT_PATH_test = '../data/bandwidth/test/mine-hit/'
    else:
        OUTPUT_PATH_train =  '../data/bandwidth/train/mine-miss/'
        OUTPUT_PATH_test = '../data/bandwidth/test/mine-miss/'

    if os.path.exists(OUTPUT_PATH_train) == False:
        os.makedirs(OUTPUT_PATH_train)
    if os.path.exists(OUTPUT_PATH_test) == False:
        os.makedirs(OUTPUT_PATH_test)

    lineCounter = 0
    traceCounter = 0
    testCounter = 0
    lastTimestamp = -1

    for ip in linesD:
        print("ip=", ip)
        lineCounter = 0
        traceCounter_perIP = 0
        lines = linesD[ip]
        while True:
            if random.random() < 0.2:
                OUTPUT_PATH = OUTPUT_PATH_test
            else:
                OUTPUT_PATH = OUTPUT_PATH_train
            output_path_ip_dir = OUTPUT_PATH + "/" + ip + "/"
            if os.path.exists(output_path_ip_dir) == False:
                os.makedirs(output_path_ip_dir)
            outputFile = open(output_path_ip_dir + str(traceCounter) + ".log", 'w')
            time = 0
            endFlag = False
            while True:
                if lineCounter >= len(lines):
                    lineCounter = 0
                    endFlag = True
                line = lines[lineCounter]
                parse = line.split("\t")
                segNum = parse[0]
                if hitFlag:
                    throughput = parse[2]
                else:
                    throughput = parse[3]
                timestamp = parse[6]
                if segNum == "1":
                    lastTimestamp = int(timestamp)
                    lineCounter += 1
                    continue
                timeInterval = int(timestamp) - lastTimestamp
                if timeInterval == 0:
                    lineCounter += 1
                    continue
                lastTimestamp = int(timestamp)
                time += timeInterval
                outputFile.write(str(time) + " " + throughput + "\n")
                lineCounter += 1
                if time > 1000:
                    time = 0
                    break
            traceCounter += 1
            traceCounter_perIP += 1
            # print("traceCounter_perIP = ", traceCounter_perIP)
            if endFlag:
                break
            if traceCounter_perIP > 30:
                break
    print("traceCounter = ", traceCounter)



def get_throughput_meanAndStd():
    bandwidthL_all = []
    for traceType in ["FCC", "HSDPA", "mine"]:
        bandwidthL = []
        filePath_1 = "../data/bandwidth/train/" + traceType
        if traceType == "mine":
            ipList = os.listdir(filePath_1)
            for ip in ipList:
                filePath = filePath_1 + "/" + ip
                for fileName in os.listdir(filePath):
                    bandwidthL.extend(np.loadtxt(filePath + "/" + fileName)[:, 1].flatten().tolist())
        else:
            for fileName in os.listdir(filePath_1):
                bandwidthL.extend(np.loadtxt(filePath_1 + "/" + fileName)[:, 1].flatten().tolist())

        mean, std = util.cal_normal_parameter(np.array(bandwidthL))
        print(traceType, "mean=", mean, "std=", std)
        bandwidthL_all.extend(bandwidthL)
    print(len(bandwidthL_all))
    mean, std = util.cal_normal_parameter(np.array(bandwidthL_all))
    print("all", "mean=", mean, "std=", std)


def main():
    # clean_rtt(fileDir="./file/rtt", fileName="rtt_raw.txt")
    # cook_fccBwTrace()
    # cook_hsdpaBwTrace()
    cook_myBwTrace(hitFlag=True)
    cook_myBwTrace(hitFlag=False)
    # get_throughput_meanAndStd()


main()