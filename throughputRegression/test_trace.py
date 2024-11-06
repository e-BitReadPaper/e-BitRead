import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas
import util

meanList= [331749.4836325781, 7843078.130064256, 1412342.5048693696, 0.8023405928554804, 1.9972304795773865, 58.37253985303123, 2521162.6465637363, 48.6385548071035, 55884.790772016444, 67029886.86461496]
stdList= [178286.95268130588, 7451465.301797467, 412573.3560083985, 1.089683410632618, 1.1034170860631605, 57.18779300160279, 270534.1982745727, 10.28260776292728, 21140.349612977632, 10944364.94569657]
eleList= ['size', 'hThroughput', 'mThroughput', 'hTime', 'mTime', 'rtt', 'maxMT', 'reqCount', 'totalMT', 'avgMT']
sampleList = ['size', 'hThroughput', 'hTime', 'rtt', 'totalMT', 'mThroughput']
sampleIndexList = []
for i in sampleList:
    sampleIndexList.append(eleList.index(i))
CUDA = 1
inputElementCount = 5
rttDict = {}

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(5, 300)
        self.hidden2 = torch.nn.Linear(300, 150)
        self.hidden3 = torch.nn.Linear(150, 150)
        self.hidden4 = torch.nn.Linear(150, 80)

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.dropout4 = torch.nn.Dropout(p=0.5)

        self.predict = torch.nn.Linear(80, 1)


    def forward(self, x):
        x = F.relu(self.dropout1(self.hidden1(x)))
        x = F.relu(self.dropout2(self.hidden2(x)))
        x = F.relu(self.dropout3(self.hidden3(x)))
        x = F.relu(self.dropout4(self.hidden4(x)))

        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


def get_rttDict():
    lines = open("../data/rtt.txt").readlines()
    rttDict = {}

    for line in lines:
        ip = line.split(" ")[0]
        rtt = line.split(" ")[1].strip()
        rttDict[ip] = rtt
    return rttDict


def draw_test(data,labels, fileName):
    plt.figure(figsize=(5, 5))
    for i in range(data.shape[1]):
        plt.plot(data[:, i]/1024/1024, label=labels[i])

    plt.legend(loc='best')
    plt.ylabel('throughput/Mbps')
    plt.title(fileName[:-4])
    plt.savefig(fileName)
    plt.close()



def gen_sample(lines,fileName):
    HISLEN = 10
    eleList = ["segNum", "size", "hThroughput", "mThroughput", "hTime", "mTime", "time", "rtt",
               "maxMT_h", "reqCount_h", "totalMT_h", "avgMT_h",
               "maxMT_m", "reqCount_m",	"totalMT_m","avgMT_m"]
    selectedEleList = ["size", "hThroughput", "mThroughput", "hTime", "mTime", "rtt", "maxMT_h", "reqCount_h", "totalMT_h", "avgMT_h"]
    selectedIndexList = []
    for i in selectedEleList:
        selectedIndexList.append(eleList.index(i))

    hisEleList = [] # history selected element list
    for i in range(len(selectedEleList)):
        hisEleList.append([])
    samplesList = []
    for i in range(len(lines)):
        line = lines[i]
        elements = line.strip().split("\t")
        skipFlag = False
        sample = []
        for j in range(len(elements)):
            if j not in selectedIndexList:
                continue
            elements[j] = float(elements[j])
            if eleList.index("rtt")==j and elements[j] == -1:
                ip = fileName.split("_")[-1][0:-4]
                if ip in rttDict:
                    elements[j] = float(rttDict[ip])
                else:
                    # print(ip)
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
            elements[j] = sum(hisEleList[selectedEleIndex])/HISLEN
            sample.append(elements[j])

        if skipFlag:
            continue

        samplesList.append(sample)
    return np.array(samplesList)


def test_neural(modelFileDir, modelIndex, data):
    data = np.array(pd.DataFrame(data).reindex(columns=sampleIndexList)) # data = [['size', 'hThroughput', 'hTime', 'rtt', 'totalMT', 'mThtoughput']]
    h = data.shape[0]
    w = data.shape[1]
    for i in range(w):
        data[:, i] = (data[:, i]-meanList[sampleIndexList[i]])/stdList[sampleIndexList[i]]
    x = torch.Tensor(data[0:h, 0:w-1])
    y = torch.Tensor(data[0:h, w-1])
    net_Adam = Net()
    if CUDA:
        net_Adam.cuda()
    net_Adam.load_state_dict(torch.load(modelFileDir+"/models/"+modelIndex+".pkl"))
    if CUDA:
        x = x.cuda()
        y = y.cuda()
    x = x.view(-1, inputElementCount)
    y = y.view(-1, 1)
    net_Adam.eval()
    output = net_Adam(x)
    pre_test_array = output.cpu().data.numpy()
    return pre_test_array * stdList[eleList.index("mThroughput")] + meanList[eleList.index("mThroughput")]


def test_linearRegression(data):
    lrModel_coef = [1.37157035e+00, 1.20239330e-02, - 6.98535635e+04, - 1.45555936e+03, 2.68264302e+00]
    lrModel_intercept = 858924.18991501
    mThroughput = np.zeros((data.shape[0],1), dtype=float)
    for i in range(len(sampleIndexList)-1):
        mThroughput = mThroughput + (data * lrModel_coef[i])[:, sampleIndexList[i]].reshape(-1, 1)
    mThroughput = mThroughput + lrModel_intercept
    return mThroughput


def test(modelFileDir, modelIndex):
    fileDir = "../../trace/throughput/trace1+2/"
    txtFileDir = modelFileDir + "/test/txt"
    pdfFileDir = modelFileDir + "/test/pdf"
    if os.path.exists(txtFileDir) == False:
        os.makedirs(txtFileDir)
    if os.path.exists(pdfFileDir) == False:
        os.makedirs(pdfFileDir)

    tLossNeural = 0
    tLossLinearRegression = 0
    sampleCount = 0
    lossListNeural = [] # cdf图
    lossListLinearRegression = [] # cdf图

    for fileName in os.listdir(fileDir):
        filePath = fileDir + fileName
        lines = open(filePath).readlines()[1:]
        data = gen_sample(lines, fileName)
        # data = [["size", "hThroughput", "mThtoughput", "hTime", "mTime", "rtt", "maxMT_h", "reqCount_h", "totalMT_h", "avgMT_h"]]
        if data.shape[0] < 10:
            continue

        mTNeural = test_neural(modelFileDir, modelIndex, data)
        mTLinearRegression = test_linearRegression(data)
        hThroughput = data[:, eleList.index("hThroughput")].reshape(-1, 1)
        mThroughput = data[:, eleList.index("mThroughput")].reshape(-1, 1)

        compareData = np.concatenate((hThroughput, mThroughput, mTNeural, mTLinearRegression), axis=1)

        lossNeural = np.sum(np.abs(mTNeural - mThroughput))
        lossLinearRegression = np.sum(np.abs(mTLinearRegression - mThroughput))
        tLossNeural += lossNeural
        tLossLinearRegression += lossLinearRegression
        sampleCount += mThroughput.shape[0]
        lossListNeural.append(lossNeural)
        lossListLinearRegression.append(lossLinearRegression)

        goodResults = ["2019-04-26_11-29-38_TDOWS26_194.29.178.14.csv",
                       "2019-04-26_12-31-33_Person10_133.69.32.133.csv",
                       "2019-04-29_03-49-33_Person4_153.90.1.35.csv",
                       "2019-04-29_04-05-21_Person4_165.242.90.128.csv"]
        if fileName in goodResults:
            np.savetxt(txtFileDir+"/"+fileName[:-4]+"_test.txt", compareData)

        # draw_test(compareData, ["hThroughput", "mThroughput", "mTNeural", "mTLinearRegression"], pdfFileDir+"/"+fileName[:-4]+"_test.pdf")

    print("neural average loss=",tLossNeural/sampleCount)
    print("linear regression loss=",tLossLinearRegression/sampleCount)
    np.savetxt(txtFileDir+"/neural.txt", np.array(lossListNeural))
    np.savetxt(txtFileDir+"/linearRegresion.txt", np.array(lossListLinearRegression))


def draw_lossCDF(modelFileDir):
    txtFileDir = modelFileDir + "/test/txt"
    lossListNeural = np.loadtxt(txtFileDir+"/neural.txt")/1024/1024
    lossListLinearRegression = np.loadtxt(txtFileDir+"/linearRegresion.txt")/1024/1024
    colorL = ["#98133a", "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9", "#A9CADB", "#d55e00"]
    markerL = ["s", "x", "o", "^", "|", "D"]
    fontSize = 12

    util.drawCDF(lossListNeural, label="neural", marker=markerL[0], color=colorL[0])
    util.drawCDF(lossListLinearRegression, label="linear regression", marker=markerL[1], color=colorL[1])
    plt.legend(loc='best', fontsize=fontSize)
    plt.xlabel("loss/Mbps")
    plt.ylabel("cdf")
    plt.show()
    plt.savefig(txtFileDir+"/lossCDF.pdf")


def main():
    global rttDict
    modelFileDir = "../data/4/model/2019-04-30_10-08-25/"
    # modelIndex = "32"
    # rttDict = get_rttDict()
    # test(modelFileDir, modelIndex)

    draw_lossCDF(modelFileDir)


main()