#from __future__ import with_statement
# -*- coding: utf-8 -*

import math
import os
import time
import threading
import sys #用于接收参数
import socket #用于获取IP地址
import random
# import ping
import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK
from requests.packages.urllib3.poolmanager import PoolManager
import numpy as np

# 网络状况设置：------------------------------------------------
rate_list = []
rate_level = 0
rate_limit = -1
network_type = -1
hit_type = 0 #hit后码率提升
http_session = requests.Session()
switch_step = -1
switch_counter = 0 # 记录距离上次切换状态多少个chunk
class_id = -1 # 由client_id决定
chainlen_probability_list = []
#--------------------------------------------------------------
IF_NO_CACHE_HEADER = False #如果为False则返回缓存中视频，否则强制miss，即使命中也不会用缓存中的视频
K = 5 # 历史数据跨度
TURN_COUNT = 1
NETWORK = 1
VIDEO_RANDOM = 1
rtt = -1 # 到源服务器rtt

#-------------------------------------
#不同种类视频SSIM值
p = [-2.1484344131667874, -1.6094379124341003, -1.0986122886681098, -0.40546510810816444, 0.0] #log(R/Rmax)
ssims = [[0.9738054275512695, 0.9835991263389587, 0.9885340929031372, 0.9929453730583191, 0.9949484467506409],
        [0.9900619387626648, 0.9940679669380188, 0.9961671233177185, 0.9980104565620422, 0.9989200830459595],
        [0.8481869101524353, 0.9067575335502625, 0.9440717697143555, 0.9729890823364258, 0.9826958775520325],
        [0.9100275039672852, 0.9230990409851074, 0.9350598454475403, 0.9525502324104309, 0.9633835554122925],
        [0.9340730905532837, 0.9563283324241638, 0.9694747924804688, 0.9818214774131775, 0.9874436259269714]]
video_type = -1 #1-5的一个值。
#-------------------------------------
segementDuration = -1 # unit:ms
bitrateSet = []
bitrateIndex = -1
last_bitrate_index = -1
segmentCount = -1
videoDuration = -1 # unit:s
segmentNum = -1 #current segment index
bufferSize = -1
throughputList = []# RB算法，计算past throughput的调和平均数
dict = {}
dict_key_list = [   'throughputList_k',
                    'downloadTime_k',
                    'chunkSize_k',
                    'ifHit_k',
                    'buffer_k',
                    'lastQoE',
                    'bitrate',
                    'rtt',
                    'chainLength',
                    'video_type',
                    'bitrate_set']


startTime = -1 #启动时间
csvFile = -1
totalRebufferTime = -1
START_BUFFER_SIZE = 8000 # When buffer is larger than 4s, video start to play.
MAX_BUFFER_SIZE = 30000 # buffer大小为8s
MIN_BUFFER_SIZE = 4000 # 如buffer小于8s，则选择最低码率
videoName = ""

originServers=["local","remote","remote2"]
originServer = originServers[0]

if originServer == "local":
    URLPrefix = "http://219.223.189.148:80/video"
    host = "219.223.189.148"
    proxy_host = "219.223.189.148"
elif originServer == "remote":
    URLPrefix = "http://video.wangyukun.com/video"
    host = "video.wangyukun.com"
    proxy_host = "video.wangyukun.com"
else:
    URLPrefix = "http://39.106.193.51/video"
    host = "39.106.193.51"
    proxy_host = "39.106.193.51"


class SourcePortAdapter(HTTPAdapter):
    """"Transport adapter" that allows us to set the source port."""

    def __init__(self, port, *args, **kwargs):
        self.poolmanager = None
        self._source_port = port
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs):
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, source_address=('', self._source_port))


def getPort():
    pscmd = "sudo netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000,20000)
    if tt not in procarr:
        return tt
    else:
        return getPort()


def savefile(filepath, filename, data):
    if os.path.exists(filepath) == False:
        os.makedirs(filepath)

    file = open(filepath + "/" + filename,'w')
    file.write(data)
    file.close()


def parseMPDFile(url):
    global segementDuration
    global bitrateSet
    global bitrateIndex
    global segmentCount
    global videoDuration

    bitrateSet = []
    lineCount = 1
    VideoStartLineCount = -1
    AudioStartLineCount = -1
    segmentCount = -1
    videoDuration = -1
    headers = {}

    if IF_NO_CACHE_HEADER == True:
        headers['Cache-Control']='no-cache'
    headers['Connection']='keep-alive'
    print("start to get mpd file")
    try:
        response = http_session.get(url, timeout=(3.05, 27), headers=headers)
    except (requests.exceptions.ConnectionError,requests.exceptions.ConnectTimeout) as err:
        print(err)
        return False

    responseStr = response.text
    status_code = str(response.status_code)
    if status_code != '200':
        print("get mpd return status code: %s" %status_code)
        return False
    # print responseStr
    mpdFileDir = "./video/" + videoName
    # print mpdFileDir
    mpdFileName = "stream.mpd"
    # print mpdFileName
    # savefile(mpdFileDir, mpdFileName, responseStr)

    lines = responseStr.split('\n')
    # print lines

    for line in lines:
        if line.find("MPD mediaPresentationDuration")!=-1:
            mediaPresentationDuration = line.split('"')[1]
            mediaPresentationDuration = mediaPresentationDuration[2:len(mediaPresentationDuration)]
            # print mediaPresentationDuration
            if mediaPresentationDuration.find("H") != -1 :
                mediaPresentationDuration_hour = int(mediaPresentationDuration.split("H")[0])
                mediaPresentationDuration_minute = int(mediaPresentationDuration.split("H")[1].split("M")[0])
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("H")[1].split("M")[1].split("S")[0])
                videoDuration = mediaPresentationDuration_hour * 3600 + mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second
            elif mediaPresentationDuration.find("M")!= -1:
                mediaPresentationDuration_minute = int(mediaPresentationDuration.split("M")[0])
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("M")[1].split("S")[0])
                videoDuration = mediaPresentationDuration_minute * 60 + mediaPresentationDuration_second

            else:
                mediaPresentationDuration_second = float(mediaPresentationDuration.split("S")[0])
                videoDuration = mediaPresentationDuration_second
            # print videoDuration

        if line.find("Video")!=-1:
            VideoStartLineCount = lineCount
        if line.find("Audio")!=-1:
            AudioStartLineCount = lineCount
        if line.find('<SegmentTemplate')!=-1 and AudioStartLineCount == -1:
            elements = line.split(' ')
            for element in elements:
                if element.startswith("duration"):
                    segementDuration = int(element.split('"')[1])
        if line.find('<Representation')!=-1 and AudioStartLineCount == -1:
            elements = line.split(' ')
            for element in elements:
                if element.startswith("bandwidth"):
                    # print(element.split(('"')))
                    bitrateSet.append(int(element.split('"')[1]))
    #bitrateIndex = bitrateSet.index(min(bitrateSet))
    segmentCount =math.ceil(videoDuration / segementDuration * 1000)
    # if network_type == 0:
    #     segmentCount = min(50,segmentCount)
    print('segmentCount: %d' %segmentCount)
    return True


def getURL(videoName,bitrateIndex,segmentNum):
    url = URLPrefix + "/" + videoName + "/video/avc1/" + str(bitrateIndex+1)+"/seg-"+str(segmentNum)+".m4s"
    print('URL: %s' %url)
    return url


def getBitrateIndex(throughput): #RB算法
    global throughputList

    if len(throughputList) < 5:
        throughputList.append(throughput)
    else:
        throughputList.append(throughput)
        throughputList.pop(0)

    reciprocal = 0 #倒数
    for i in range(len(throughputList)):
        reciprocal += 1/throughputList[i]
    reciprocal /= len(throughputList)

    if reciprocal!=0:
        throughputHarmonic = 1/reciprocal
    else:
        throughputHarmonic = 0

    # print("throughput harmonic: %f" % throughputHarmonic)

    for i in range(len(bitrateSet)):
        if throughputHarmonic < bitrateSet[i]:
            if i-1 < 0:
                return i
            else:
                return i-1

    return len(bitrateSet)-1

def dict2str(dict):
    s = ""

    for key in dict_key_list:
        for item in dict[key]:
            s+=str(item)+" "
    s.strip()
    return s


def startup():
    global segmentNum
    global bufferSize
    global bitrateIndex
    global last_bitrate_index
    global videoName
    global csvFile
    global dict
    global switch_counter
    global rate_level
    global startTime

    while bufferSize < START_BUFFER_SIZE:
        ifHit = -1
        startDownloadTime = time.time()
        url = getURL(videoName,bitrateIndex,segmentNum)
        #header------------------------------------------
        headers = {}
        headers['Connection'] = 'keep-alive'
        headers['X-Debug'] = 'X-Cache'
        prefetch_header = dict2str(dict) # 如果是第一块，这个dict的value都是空的。len(prefetch_header)=0
        if len(prefetch_header) != 0:
            headers['Prefetch'] = prefetch_header
        if IF_NO_CACHE_HEADER == True:  # 强制miss
            headers['Cache-Control'] = 'no-cache'
        headers['bitrate'] = str(bitrateSet[bitrateIndex]/1024)
        #header------------------------------------------
        try:
            response = http_session.get(url, timeout=(3.05, 27), headers=headers)
            contentLength = float(response.headers['content-length'])  # 单位：B
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as err:
            print(err)
            return False
        endDownloadTime = time.time() #单位：s

        if ("X-Cache" in response.headers and response.headers["X-Cache"].find("miss") != -1) or IF_NO_CACHE_HEADER == True:  # 如果X-Cache中有“Miss” 或者 强制miss
            ifHit = 0
        else:
            ifHit = 1

        bufferSize += segementDuration # 单位ms
        downloadTime = endDownloadTime - startDownloadTime #单位s
        startTime += downloadTime
        throughput = contentLength*8/downloadTime # 单位bps

        if last_bitrate_index == -1:
            qualityVariation = abs(0 - ssims[video_type-1][bitrateIndex])
        else:
            qualityVariation = abs(ssims[video_type-1][last_bitrate_index] - ssims[video_type-1][bitrateIndex])

        qoe = ssims[video_type-1][bitrateIndex]-qualityVariation - downloadTime
        if len(dict['chainLength'])==0:
            dict['chainLength'] = [1]
        timestamp = int(time.time())
        ss = "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d" % (
            segmentNum, contentLength, ifHit, bufferSize, bitrateSet[bitrateIndex], throughput, downloadTime,
            downloadTime, video_type, rate_list[rate_level], hit_type, network_type, dict['chainLength'][-1], qoe, timestamp)
        csvFile.write(ss + '\n')
        print("segNum\tchunkS\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\tvtype\trate\thitty\tnet\tchainL\tqoe\ttime")
        print(ss)

        last_bitrate_index = bitrateIndex
        bitrateIndex = getBitrateIndex(throughput)
        # 要加到请求中的信息----------------------
        dict['bitrate_set'] = bitrateSet
        if last_bitrate_index == bitrateIndex:
            dict['chainLength'][0] += 1
        else:
            dict['chainLength']=[1]
        if len(dict['downloadTime_k']) < K:
            dict['downloadTime_k'] = [downloadTime] * K
            dict['chunkSize_k'] = [contentLength] * K
            dict['throughputList_k'] = [throughput] * K
            dict['ifHit_k'] = [ifHit] * K
            dict['buffer_k'] = [bufferSize] * K
        else:
            dict['downloadTime_k'] = dict['downloadTime_k'][1: K] + [downloadTime]
            dict['chunkSize_k'] = dict['chunkSize_k'][1: K] + [contentLength]
            dict['throughputList_k'] = dict['throughputList_k'][1: K] + [throughput]
            dict['ifHit_k'] = dict['ifHit_k'][1: K] + [ifHit]
            dict['buffer_k'] = dict['buffer_k'][1: K] + [bufferSize]

        dict['lastQoE'] = [qoe]
        dict['bitrate'] = [str(bitrateSet[bitrateIndex])]
        dict['rtt'] =[rtt]  # 暂时还不知道咋算
        dict['video_type'] = [video_type]
        # 要加到请求中的信息----------------------

        segmentNum = segmentNum + 1
        if segmentNum > segmentCount:
            return True


def download():
    global segmentNum
    global bufferSize
    global bitrateIndex
    global last_bitrate_index
    global videoName
    global csvFile
    global totalRebufferTime
    global dict
    global switch_counter
    global switch_step
    global rate_level

    while True:
        if bufferSize + segementDuration <= MAX_BUFFER_SIZE:  # download new segment
            ifHit = -1
            startDownloadTime = time.time() # 单位 s
            url = getURL(videoName, bitrateIndex, segmentNum)
            # header------------------------------------------
            headers = {}
            headers['Connection'] = 'keep-alive'
            headers['X-Debug'] = 'X-Cache'
            prefetch_header = dict2str(dict)
            headers['Prefetch']= prefetch_header
            if IF_NO_CACHE_HEADER == True: # 强制miss
                headers['Cache-Control']= 'no-cache'
            headers['bitrate'] = str(bitrateSet[bitrateIndex]/1024)
            # header------------------------------------------
            try:
                response = http_session.get(url, timeout=(3.05, 27), headers=headers)
                contentLength = float(response.headers['content-length'])  # 单位：B
            except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as err:
                print(err)
                return False
            endDownloadTime = time.time() # 单位 s
            if "X-Cache" in response.headers:
                # print(response.headers)
                if response.headers["X-Cache"].find("miss") != -1 or IF_NO_CACHE_HEADER == True: #如果X-Cache中有“Miss” 或者 强制miss
                    ifHit = 0
                else:
                    ifHit = 1
            downloadTime = endDownloadTime - startDownloadTime #单位 s
            #--------------------------------------------------
            # 由于播放，buffer减少，若buffer减少至0则卡顿
            rebufferTimeOneSeg = -1
            if downloadTime*1000 > bufferSize:
                rebufferTimeOneSeg = downloadTime - bufferSize/1000 #单位 s
                bufferSize = 0
                totalRebufferTime += rebufferTimeOneSeg
            else:
                bufferSize = bufferSize - downloadTime*1000 #单位 ms
                rebufferTimeOneSeg = 0
            bufferSize += segementDuration
            #--------------------------------------------------
            throughput = contentLength*8 / downloadTime  # 单位bps
            qualityVariation = abs(ssims[video_type-1][last_bitrate_index] - ssims[video_type-1][bitrateIndex])
            qoe = ssims[video_type-1][bitrateIndex] - qualityVariation - rebufferTimeOneSeg
            #---------------------------------------------------
            timestamp = int(time.time())
            ss = "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d" % (
                segmentNum, contentLength, ifHit, bufferSize, bitrateSet[bitrateIndex], throughput, downloadTime,
                rebufferTimeOneSeg, video_type, rate_list[rate_level], hit_type, network_type, dict['chainLength'][-1], qoe, timestamp)
            csvFile.write(ss + '\n')
            csvFile.flush()
            print(
                "segNum\tchunkS\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\tvtype\trate\thitty\tnet\tchainL\tqoe\ttime")
            print(ss)
            #---------------------------------------------------
            last_bitrate_index = bitrateIndex
            if bufferSize < MIN_BUFFER_SIZE:
                bitrateIndex = 0
            else:
                bitrateIndex = getBitrateIndex(throughput)

            # 要加到请求中的信息----------------------------------------
            dict['bitrate_set'] = bitrateSet
            if last_bitrate_index == bitrateIndex:
                dict['chainLength'][0] += 1
            else:
                dict['chainLength']=[1]
            if len(dict['downloadTime_k']) < K:
                dict['downloadTime_k'] = [downloadTime] * K
                dict['chunkSize_k'] = [contentLength] * K
                dict['throughputList_k'] = [throughput] * K
                dict['ifHit_k'] = [ifHit] * K
                dict['buffer_k'] = [bufferSize] * K
            else:
                dict['downloadTime_k'] = dict['downloadTime_k'][1: K] + [downloadTime]
                dict['chunkSize_k'] = dict['chunkSize_k'][1: K] + [contentLength]
                dict['throughputList_k'] = dict['throughputList_k'][1: K] + [throughput]
                dict['ifHit_k'] = dict['ifHit_k'][1: K] + [ifHit]
                dict['buffer_k'] = dict['buffer_k'][1: K] + [bufferSize]

            dict['lastQoE'] = [qoe]
            dict['bitrate'] = [str(bitrateSet[bitrateIndex])]
            dict['rtt'] = [rtt]  # 暂时还不知道咋算
            dict['video_type'] = [video_type]

            # 切换网络状态---------------------------
            if NETWORK and network_type == 2:
                switch_counter += 1
                if switch_counter == switch_step:
                    rand_up = random.random()
                    if rand_up < 0.4:
                        rate_level = rate_level + 1
                    elif rand_up >0.6:
                        rate_level = rate_level - 1
                    else:
                        rate_level = rate_level

                    if rate_level < 0:
                        rate_level = rate_level + 2
                    if rate_level >= len(rate_list):
                        rate_level = rate_level-2
                    rate_limit = rate_list[rate_level]
                    print("new rate_limit=", rate_limit)
                    cmd = "sudo tc class change dev ifb0 parent 1:1 classid 1:{} htb rate {}Kbit".format(class_id, rate_limit/1024)
                    os.popen(cmd)
                    switch_counter = 0
                    mu, sigma = 2.8, 0.5  # mean and standard deviation
                    s = list(np.random.lognormal(mu, sigma, 1000))
                    switch_step = int(random.sample(s,1)[0])
                    if switch_step < 1:
                        switch_step = 1
                    print("new switch step =", switch_step)
                    time.sleep(1)
            # 切换网络状态---------------------------

            segmentNum = segmentNum + 1
            if segmentNum > segmentCount:
                break
        else:
            bufferSize = MAX_BUFFER_SIZE - segementDuration
    return True

def main():
    global segementDuration
    global bitrateSet
    global bitrateIndex
    global last_bitrate_index
    global segmentCount
    global videoDuration
    global segmentNum
    global bufferSize
    global throughputList
    global startTime
    global csvFile
    global totalRebufferTime
    global videoName
    global TURN_COUNT
    global IF_NO_CACHE_HEADER
    global class_id
    global video_type
    global hit_type
    global rtt
    global rate_level
    global rate_list
    global switch_step
    global dict
    global network_type

    traceIndex = 0 # 区分每次运行结果
    client_id = 1
    network_type = -1
    TURN_COUNT = 1
    #命令行参数
    for i in range(1, len(sys.argv)):
        # print("参数", i, sys.argv[i])
        if i == 1:
            network_type = int(sys.argv[i])
        if i == 2:
            traceIndex = int(sys.argv[i])
        if i == 3:
            client_id = int(sys.argv[i])
        if i == 4:
            TURN_COUNT = int(sys.argv[i])

    csvFileDir = "./trace/trace"+str(traceIndex)

    #ping to get rtt--------------------------
    # p= ping.ICMPPing()
    # rtt = p.main(proxy_host)
    #--------------------------------------------------------------
    # 获取视频列表和流行度
    video_popularity_file = open("./video.txt")
    video_popularity_list = video_popularity_file.readlines()
    video_popularity_list = [(i.split(" ")[0],float(i.split(" ")[1]),int(float(i.split(" ")[2]))) for i in video_popularity_list] #(video_name, popularity, video_tupe)
    #---------------------------------------------------------------
    # 绑定端口号-----------------------------------------------------
    port_number = getPort() #对于一个client来说他的端口号固定。
    print("port_number=", port_number)
    http_session.mount('http://' + host + ":80", SourcePortAdapter(port_number))
    # ---------------------------------------------------------------
    network_initial_flag = False
    for turn in range(TURN_COUNT):
        if os.path.exists(csvFileDir) == False:
            os.makedirs(csvFileDir)
        for i in range(1):
            # 初始化变量
            segementDuration = -1  # unit:ms
            bitrateSet = []
            bitrateIndex = 0
            last_bitrate_index = -1
            segmentCount = -1
            videoDuration = -1  # unit:s
            segmentNum = 1  # current segment index
            bufferSize = 0
            throughputList = []  # RB算法，计算past throughput的调和平均数
            startTime = -1  # 启动时间
            csvFile = -1
            totalRebufferTime = 0
            videoName = ""
            video_type = 0
            rtt = -1
            rate_level = 0
            dict = {'throughputList_k': [],
                    'downloadTime_k': [],
                    'chunkSize_k': [],
                    'ifHit_k': [],
                    'buffer_k':[],
                    'lastQoE': [],
                    'bitrate': [],
                    'rtt': [],
                    'chainLength': [],
                    'video_type':[],
                    'bitrate_set':[]}
            # ---------------------------------------------------------------
            if VIDEO_RANDOM:
                video_random = random.random()
                videoName = ""
                for i in range(len(video_popularity_list)):
                    if video_random < video_popularity_list[i][1]:
                        videoName = video_popularity_list[i-1][0]
                        video_type = video_popularity_list[i-1][2]
                        break
                if videoName == "":
                    videoName = video_popularity_list[-1][0]
                    video_type = 1
            else:
                videoName = "yourname"
                video_type = 1
            mpd_url = URLPrefix+"/"+videoName+"/stream.mpd"
            print('video name=',videoName)
            ifSuccess = parseMPDFile(mpd_url)
            if ifSuccess == False:
                print("parse MPD file failed")
                return
            # ---------------------------------------------------------------
            if NETWORK:
                # 设置网络条件
                p = [0.1,0.8,0.1]
                p_sum = []
                sum = 0
                for pi in p:
                    sum += pi
                    p_sum.append(sum)

                if network_type == -1:
                    net_rand = random.random()
                    print("net_rand=",net_rand )
                    for i in range(len(p_sum)):
                        if net_rand < p_sum[i]:
                            network_type = i+1
                            break

                print("network_type = ",network_type)

                if network_type == 1: #一直卡顿
                    rate_list = [bitrateSet[0] - 50*1024]
                    hit_type = 1
                elif network_type == 2:
                    hit_type = random.randint(0, 1)
                    if hit_type == 0: #prefetch有意义
                        rate_list = [bitrateSet[0] - 50*1024, bitrateSet[0] + 30*1024, bitrateSet[1] + 50*1024, bitrateSet[2]+150*1024,bitrateSet[3] + 150*1024,bitrateSet[4] + 150*1024]
                    else: #prefetch没意义
                        rate_list = [bitrateSet[1] - 50*1024, bitrateSet[2] - 100*1024,bitrateSet[3] -100*1024, bitrateSet[4] -100*1024,bitrateSet[4] +100*1024]
                    switch_step = 10
                    rate_level = int(len(rate_list)/2)

                elif network_type == 3:
                    rate_list = [bitrateSet[4] +1000*1024]
                    hit_type = 1
                else:
                    network_type = 2
                    hit_type = random.randint(0, 1)
                    if hit_type == 0: #prefetch有意义
                        rate_list = [bitrateSet[0] - 50*1024, bitrateSet[0] + 30*1024, bitrateSet[1] + 50*1024,
                                     bitrateSet[2]+50*1024,bitrateSet[3] + 50*1024,bitrateSet[4] + 50*1024]
                    else: #prefetch没意义
                        rate_list = [bitrateSet[1] - 50*1024, bitrateSet[2] - 100*1024,bitrateSet[3] -100*1024,
                                     bitrateSet[4] -100*1024,bitrateSet[4] +100*1024]
                    switch_step = 10


                rate_limit = rate_list[rate_level]
                print("rate_limit=", rate_limit)
                class_id = client_id * 10

                if network_initial_flag == False:
                    cmd1 = "sudo tc class add dev ifb0 parent 1:1 classid 1:{} htb rate {}Kbit".format(class_id, rate_limit/1024)
                    cmd2 = "sudo tc filter add dev ifb0 parent 1:0 protocol ip prio 100 u32 match ip dport {} 0xffff flowid 1:{}".format(port_number, class_id)
                    os.popen(cmd1)
                    os.popen(cmd2)
                    network_initial_flag = True
                else:
                    rate_limit = rate_list[rate_level]
                    cmd = "sudo tc class change dev ifb0 parent 1:1 classid 1:{} htb rate {}Kbit".format(class_id, rate_limit / 1024)
                    os.popen(cmd)
                time.sleep(1)
            # ---------------------------------------------------------------
            time_now = int(time.time())
            time_local = time.localtime(time_now)
            dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
            ran = random.random()
            csvFileName = csvFileDir +"/"+ videoName +"_"+ str(dt) +"_"+ str(client_id)+ ".csv"
            csvFile = open(csvFileName, 'w')
            csvFile.write("segNum\tsize\tHit\tbuffer\tbitrate\tthroughput\tdownloadT\trebufferT\t"
                  "vtype\tratelimit\thittype\tnetwork\tchainLen\tqoe\ttime\n")
            startTime = 0
            ifSuccess = startup()
            if ifSuccess==False:
                print("startup failed")
                return
            startupLatency = startTime
            ifSuccess = download()
            if ifSuccess==False:
                print("download failed")
                return
            csvFile.close()
            print("play complete")
            print("startupLatency=" + str(startupLatency))
            print("total rebuffering time=" + str(totalRebufferTime) +"ms")

main()