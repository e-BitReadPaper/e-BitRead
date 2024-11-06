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


BATCH_SIZE = 50
LR = 0.01
EPOCH = 200
decay_rate = 0.2
CUDA=1
inputElementCount = 5



class MyDataset(Data.Dataset):
    def __init__(self, data_dir):
        meanList = [341390.70424187323, 3287088.0509502394, 2219683.7490228005, 1.2997353281592816, 1.6118756550731155,
                    61.88723814098079, 5565846.583892792, 52.64681558597572, 109102.86260377312, 115701676.1543384]
        stdList = [183878.4092394808, 1972978.4135890864, 1149431.6002636002, 1.6760607217266807, 1.677710308928094,
                   67.75019604650828, 1108736.4209206265, 6.7569834378213365, 32181.318236872674, 9448114.651317576]
        eleList = ['size', 'hThroughput', 'mThroughput', 'hTime', 'mTime', 'rtt', 'maxMT', 'reqCount', 'totalMT',
                   'avgMT']
        sampleList = ['size', 'hThroughput', 'hTime', 'rtt', 'totalMT', 'mThroughput']
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

#训练数据格式：throughputs + segment_sizes + lastQoE + bufferSize + bitrate , QoEs

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


        # x = F.relu(self.dropout1(self.hidden1(x)))
        # x = F.relu(self.dropout2(self.hidden2(x)))
        # x = F.relu(self.dropout3(self.hidden3(x)))
        # x = F.relu(self.dropout4(self.hidden4(x)))

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


def main():
    dataIndex = "1"
    dir = "../../../data/throughputRelation/data/"+dataIndex
    trainingDataDir =  dir + "/train"
    testDataDir = dir + "/test"

    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)

    modelFileDir = dir + "/DNNmodel/" + dt

    if os.path.exists(modelFileDir) == False:
        os.makedirs(modelFileDir)

    # different nets
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
    figureNum = 1
    train_dataset = MyDataset(trainingDataDir)
    test_dataset = MyDataset(testDataDir)
    LR_ = LR
    for epoch in range(EPOCH):
        f_loss = open(lossFileName, 'a')
        print("epoch ",epoch)

        if epoch%20 == 0:
            LR_ = LR*decay_rate
            if LR_ > 0.001:
                for param_group in opt_Adam.param_groups:
                    param_group['lr'] = param_group['lr'] * decay_rate

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
            b_x = b_x.view(-1, inputElementCount)
            b_y = b_y.view(-1, 1)
            net_Adam.train()
            output = net_Adam(b_x)
            loss = loss_func(output, b_y)

            opt_Adam.zero_grad()
            loss.backward()
            opt_Adam.step()

            loss_train_list.extend(torch.abs(output.cpu().data - b_y_cpu.data).numpy().flatten())  # loss recoder
            if epoch % 20 == 0:
                pre_mthr_train_list.extend(output.cpu().data.numpy().flatten().tolist())
                real_mthr_train_list.extend(b_y_cpu.cpu().data.numpy().flatten().tolist())

        loss_avg_train = np.mean(np.array(loss_train_list))
        loss_avg_train_list.append(loss_avg_train)

        print("train: Epoch"+str(epoch)+" loss: "+str(loss_avg_train))

        # if epoch % 20 == 0:
        #     pre_train_array=np.array(pre_mthr_train_list).reshape(len(pre_mthr_train_list),1)
        #     real_train_array=np.array(real_mthr_train_list).reshape(len(real_mthr_train_list),1)
        #     compare_array=np.concatenate((pre_train_array,real_train_array), axis=1)
        #
        #     path = modelFileDir + "/compare"
        #     if os.path.exists(path) == False:
        #         os.makedirs(path)
        #     np.savetxt(path+"/train_qoe_"+str(epoch)+".txt", compare_array)

        #test----------------------------------------------------
        if CUDA:
            x = test_dataset.x.cuda()
            y = test_dataset.y.cuda()
        else:
            x = test_dataset.x
            y = test_dataset.y

        x = x.view(-1, inputElementCount)
        y = y.view(-1, 1)

        net_Adam.eval()
        output = net_Adam(x)
        loss = loss_func(output, y)

        if epoch % 20 == 0:
            pre_test_array = output.cpu().data.numpy()
            real_test_array= y.cpu().data.numpy()
            compare_array = np.concatenate((pre_test_array,real_test_array), axis=1)
            path = modelFileDir + "/compare"
            if os.path.exists(path) == False:
                os.makedirs(path)
            np.savetxt(path+"/test_qoe_"+str(epoch)+".txt", compare_array)

        loss_test_array=torch.abs(output.cpu().data - y.cpu().data).numpy()
        loss_avg_test = np.mean(loss_test_array)
        loss_avg_test_list.append(loss_avg_test)

        print("test: Epoch"+str(epoch)+" loss:"+str(loss_avg_test))
        #test end-----------------------------------------------------

        f_loss.write(str(epoch)+" "+str(loss_avg_train)+" "+str(loss_avg_test)+"\n")

        #save model
        if loss_avg_min_test > loss_avg_test or loss_avg_min_train > loss_avg_train:
            if loss_avg_min_test > loss_avg_test:
                loss_avg_min_test = loss_avg_test
            if loss_avg_min_train > loss_avg_train:
                loss_avg_min_train = loss_avg_train
            torch.save(net_Adam.state_dict(), model_dir + "/" + str(epoch) + ".pkl")

    labels = ['Adam_train', 'Adam_test']
    plt.figure(num=figureNum,figsize=(10,5))
    plt.subplot(121)
    plt.plot(loss_avg_train_list, label=labels[0])
    plt.subplot(122)
    plt.plot(loss_avg_test_list, label=labels[1])

    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(modelFileDir+"/loss.jpg")


if __name__ == '__main__':
    main()




