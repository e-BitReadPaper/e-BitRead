import matplotlib.pyplot as plt
import numpy as np
import os

DIR = ""

def compare(x, y):
    stat_x = os.stat(DIR + "/" + x)
    stat_y = os.stat(DIR + "/" + y)

    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0

def plot_train_res():
    data = np.loadtxt("../../data/RL_model/2019-06-28_10-18-56/res.txt").flatten()
    # res = []
    # temp = data[0]
    # for i in range(data.size):
    #     res.append(temp)
    #     temp = temp * 0.99 + data[i] * 0.01
    #
    plt.figure(num=1,figsize=(10,5))
    plt.plot(data)
    plt.show()
    #
    # plt.figure(num=2,figsize=(10,5))
    # resTurn = []
    # resTurns = []
    # for i in range(data.size):
    #     if len(resTurn) < 1000:
    #         resTurn.append(data[i])
    #     else:
    #         resTurn = resTurn[1:] + [data[i]]
    #     resTurns.append(sum(resTurn) / len(resTurn))
    #
    # plt.plot(resTurns)
    # plt.show()

    res_L = data.tolist()
    maximum = max(res_L)
    print("max=", maximum)
    print("epoch=", res_L.index(maximum))

    # top_k_idx=res.argsort()[::-1][0:100]
    #
    # for i in top_k_idx:
    #     print("reward = ", res[i])
    #     print("epoch = ", i)


def plot_test_res():
    traceIndex = "1"
    dir = "../../data/RL_model/2019-06-28_10-18-56/test_res_"+traceIndex
    for fileName in os.listdir(dir):
        data = np.loadtxt(dir + "/" + fileName)
        plt.plot(data, label=fileName)
        print(fileName,"=", np.average(data))

    plt.legend(loc='best')
    plt.show()

    # fileNameL = os.listdir("../../data/RL_model/2019-06-18_14-35-47/test_trace_"+traceIndex+"/mine/RL")
    # len = min(data_1.size, data_2.size)
    # for i in range(len):
    #     if data_1[i] < data_2[i]:
    #         print(i + 1, fileNameL[i])

    # for i in


def main():
    # plot_train_res()
    plot_test_res()


main()