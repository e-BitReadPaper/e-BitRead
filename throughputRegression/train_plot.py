import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

mThroughput_mean = 2219683.7490228005
mThroughput_std = 1149431.6002636002


def compare(epoch=0, flag="train"):
    if flag == "train":
        data = np.loadtxt(dir + "compare/train_qoe_"+str(epoch)+".txt")
    else:
        data = np.loadtxt(dir + "compare/test_qoe_"+str(epoch)+".txt")

    data=(data*mThroughput_std+mThroughput_mean)/1024/1024
    x = range(data.shape[0])
    plt.figure(figsize=(5,5))
    plt.scatter(data[:, 0],data[:, 1],s=5,color='green',alpha=0.1)
    plt.plot([data[:, 1].min(), data[:, 1].max()], [data[:, 1].min(), data[:, 1].max()], 'k--', lw=1,color='green')
    plt.xlabel("predict")
    plt.ylabel("real")
    plt.title(flag+"-"+str(epoch))
    plt.savefig(dir+ "/" +flag+"-"+str(epoch)+".pdf")
    plt.show()

def loss(dir):
    data = np.loadtxt(dir + "loss.txt")
    x = data[:, 0]
    train_loss = data[:, 1]
    test_loss = data[:, 2]

    epoch = np.argmin(test_loss, axis=0)
    print("min loss epoch= %d, min loss= %f" % (epoch, np.min(test_loss)))

    fig_loss = plt.figure(figsize=(7.5, 3))
    train_loss = (train_loss * mThroughput_std + mThroughput_mean)/1024/1024
    test_loss = (test_loss * mThroughput_std + mThroughput_mean)/1024/1024
    # print(train_loss.tolist())
    # print(test_loss.tolist())
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.legend(loc='best')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(dir+"loss.pdf")
    plt.show()


def main():
    dir = "../../data/throughputRelation/data/2/h2m/DNNmodel/2019-06-17_14-41-37/"
    loss(dir)
    # compare(epoch=20, flag="test")

main()