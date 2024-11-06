# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.init as Init
import sys
import time
import pandas as pd

RTT_FLAG = True
if RTT_FLAG:
    S_LEN = 4  # 'size', 'hThroughput', 'hTime', 'rtt'
else:
    S_LEN = 3  # 'size', 'hThroughput', 'hTime'

BATCH_SIZE = 20     # 批训练的数据个数
LR = 0.01
EPOCH = 1000
CUDA=1



class MyDataset(Data.Dataset):
    def __init__(self, data_dir):

        meanList = [356323.17810702475, 4284747.95493097, 1635876.4524747306, 1.279115388724093, 2.072012292996366,
                    74.95483943816453, 4540825.0233448455, 37.296557078554066, 62860381.14042809, 144473.52233441724]
        stdList = [214823.0118034271, 3844488.137285077, 848096.8208893111, 1.3002579292457577, 1.2530051518669487,
                   68.03256521873818, 866728.5593723704, 12.018775621735784, 21209161.496662676, 88430.02046708291]
        eleList = ['size', 'hThroughput', 'mThroughput', 'hTime', 'mTime', 'rtt', 'maxMT', 'reqCount', 'totalMT',
                   'avgMT']
        if RTT_FLAG == True:
            sampleList = ['size', 'hThroughput', 'hTime', 'rtt', 'mThroughput']
        else:
            sampleList = ['size', 'hThroughput', 'hTime', 'mThroughput']

        sampleIndexList = []
        for i in sampleList:
            sampleIndexList.append(eleList.index(i))

        self.x = torch.FloatTensor()
        self.y = torch.FloatTensor()

        for file_name in os.listdir(data_dir):
            data = np.loadtxt(data_dir+"/"+file_name, skiprows=1)
            data = np.array(pd.DataFrame(data).reindex(columns=sampleIndexList))
            h = data.shape[0]
            w = data.shape[1]
            for i in range(w):
                data[:, i] = (data[:, i] - meanList[sampleIndexList[i]]) / stdList[sampleIndexList[i]]
            self.x = torch.cat((self.x, torch.Tensor(data[0:h, 0:w - 1])), 0)
            self.y = torch.cat((self.y,torch.Tensor(data[0:h, w - 1])), 0)


    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.x.size(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    if classname.find("Linear") != -1:
        Init.xavier_uniform_(m.weight.data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(S_LEN, 100)
        self.hidden2 = torch.nn.Linear(100, 50)
        self.predict = torch.nn.Linear(50, 1)


    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


def train():
    dataIndex = "2"
    dir = "../../../data/throughputRelation/data/"+dataIndex
    trainingDataDir =  dir + "/train"
    testDataDir = dir + "/test"
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
    modelFileDir = dir + "/DNNmodel/" + dt
    if os.path.exists(modelFileDir) == False:
        os.makedirs(modelFileDir)

    net_Adam = Net()
    if CUDA:
        net_Adam.cuda()
    net_Adam.apply(weights_init)
    # net_Adam.load_state_dict(torch.load('../data/4/model/2019-04-30_10-08-25/models/32.pkl'))

    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()
    if CUDA:
        loss_func.cuda()

    loss_avg_train_list = []
    loss_avg_test_list = []
    loss_avg_min_train = 1000000
    loss_avg_min_test = 1000000
    lossFileName = modelFileDir + "/" + "loss.txt"
    model_dir = modelFileDir+"/"+"models"
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    train_dataset = MyDataset(trainingDataDir)
    test_dataset = MyDataset(testDataDir)
    for epoch in range(EPOCH):
        f_loss = open(lossFileName, 'a')
        print("epoch ",epoch)
        # training------------------------------------------------
        loader = Data.DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,
            num_workers=2,
        )
        loss_train_list = []
        pre_mthr_train_list = []
        real_mthr_train_list = []
        for step, (b_x_cpu, b_y_cpu) in enumerate(loader):  # for each training step
            # train your data...
            if CUDA:
                b_x = b_x_cpu.cuda()
                b_y = b_y_cpu.cuda()
            else:
                b_x = b_x_cpu
                b_y = b_y_cpu
            b_x = b_x.view(-1, S_LEN)
            b_y = b_y.view(-1, 1)
            net_Adam.train()
            output = net_Adam(b_x)
            loss = loss_func(output, b_y)
            opt_Adam.zero_grad()
            loss.backward()
            opt_Adam.step()
            loss_train_list.append(loss.cpu().data.numpy())  # loss recoder

        loss_avg_train = np.mean(loss_train_list)
        loss_avg_train_list.append(loss_avg_train)
        print("train: Epoch"+str(epoch)+" loss: "+str(loss_avg_train))
        #test----------------------------------------------------
        if CUDA:
            x = test_dataset.x.cuda()
            y = test_dataset.y.cuda()
        else:
            x = test_dataset.x
            y = test_dataset.y
        x = x.view(-1, S_LEN)
        y = y.view(-1, 1)
        net_Adam.eval()
        output = net_Adam(x)
        loss = loss_func(output, y)
        loss_avg_test = loss.cpu().data.numpy()
        loss_avg_test_list.append(loss_avg_test)

        print("test: Epoch"+str(epoch)+" loss:"+str(loss_avg_test))
        #test end-----------------------------------------------------
        f_loss.write(str(epoch) + " " + str(loss_avg_train) + " " + str(loss_avg_test) + "\n")
        #save model
        if loss_avg_min_test > loss_avg_test or loss_avg_min_train > loss_avg_train or epoch % 10 == 0:
            if loss_avg_min_test > loss_avg_test:
                loss_avg_min_test = loss_avg_test
            if loss_avg_min_train > loss_avg_train:
                loss_avg_min_train = loss_avg_train
            torch.save(net_Adam.state_dict(), model_dir + "/" + str(epoch) + ".pkl")

    labels = ['Adam_train', 'Adam_test']
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(loss_avg_train_list, label=labels[0])
    plt.subplot(122)
    plt.plot(loss_avg_test_list, label=labels[1])

    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(modelFileDir+"/loss.jpg")


def test(dataDir, modelFile):
    testDataDir = dataDir
    net_Adam = Net()
    if CUDA:
        net_Adam.cuda()
    net_Adam.load_state_dict(torch.load(modelFile))

    loss_avg_test_list = []
    loss_avg_min_test = 1000000
    test_dataset = MyDataset(testDataDir)
    #test----------------------------------------------------
    if CUDA:
        x = test_dataset.x.cuda()
        y = test_dataset.y.cuda()
    else:
        x = test_dataset.x
        y = test_dataset.y

    x = x.view(-1, S_LEN)
    y = y.view(-1, 1)

    net_Adam.eval()
    startTime = time.time()
    output = net_Adam(x)
    endTime = time.time()
    loss_test_array=(output.cpu().data - y.cpu().data).numpy()
    loss_avg_test = np.mean(np.square(loss_test_array))
    print("loss_avg_test=", loss_avg_test, "testTime=", endTime - startTime)


def main():
    global RTT_FLAG

    # train()

    # RTT_FLAG = False
    # test(dataDir="../../../data/throughputRelation/data/2/test",
    #      modelFile="../../../data/throughputRelation/data/2/DNNmodel/2019-06-17_14-41-37/models/17.pkl")

    RTT_FLAG = True
    test(dataDir="../../../data/throughputRelation/data/2/test",
         modelFile="../../../data/throughputRelation/data/2/DNNmodel/2019-06-15_17-44-28/models/554.pkl")


if __name__ == '__main__':
    main()




