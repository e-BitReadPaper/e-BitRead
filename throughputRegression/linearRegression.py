import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*
import matplotlib
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def regression(hit_to_miss_flag, rtt_flag):
    # data = pandas.read_csv(dir_prefix+"/sample.txt", sep=" ")
    # font = {'family': 'SimHei'}
    # matplotlib.rc('font', **font)
    # data[["size", "hThroughput", "mThroughput", "hTime", "mTime", "rtt", "maxMT", "reqCount", "totalMT", "avgMT"]].corr()

    # if hit_to_miss_flag==1:
    #     X = data[["size", "hThroughput", "hTime", "rtt", "totalMT"]]
    #     y = data[["mThroughput"]]
    # else:
    #     X = data[["size", "mThroughput", "mTime", "rtt", "totalMT"]]
    #     y = data[["hThroughput"]]
    train = pd.read_csv("../../data/throughputRelation/data/2/train/sample.csv", delimiter=' ')
    test = pd.read_csv("../../data/throughputRelation/data/2/test/sample.csv", delimiter=' ')
    if rtt_flag:
        train.drop(['mTime', 'maxMT', 'reqCount','totalMT', 'avgMT'], axis=1, inplace=True)
        test.drop(['mTime', 'maxMT', 'reqCount', 'totalMT', 'avgMT'], axis=1, inplace=True)
    else:
        train.drop(['mTime', 'maxMT', 'reqCount','totalMT', 'avgMT', 'rtt'], axis=1, inplace=True)
        test.drop(['mTime', 'maxMT', 'reqCount', 'totalMT', 'avgMT', 'rtt'], axis=1, inplace=True)


    if hit_to_miss_flag:
        x_train = train.drop('mThroughput', axis=1)
        y_train = train['mThroughput']
        x_test = test.drop('mThroughput', axis=1)
        y_test = test['mThroughput']/1024/1024
    else:
        x_train = train.drop('hThroughput', axis=1)
        y_train = train['hThroughput']
        x_test = test.drop('hThroughput', axis=1)
        y_test = test['hThroughput']/1024/1024

    print(x_train.columns)

    lrModel = LinearRegression()
    lrModel.fit(x_train, y_train)
    predicted = lrModel.predict(x_test)/1024/1024

    print("lrModel.coef_=", lrModel.coef_)
    print("lrModel.intercept_", lrModel.intercept_)
    # plt.rcParams['figure.figsize'] = (4.0, 4.0)
    # fig, ax = plt.subplots()
    # ax.scatter(y_test, predicted, s=2, alpha=0.1)
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.savefig(dir_prefix+"LinearRegression" + str(hit_to_miss_flag)+ ".pdf")
    # plt.show()

    avg_loss = np.average(np.square(predicted - y_test))
    print(avg_loss)





def main():
    # regression(hit_to_miss_flag=True, rtt_flag=True)
    # regression(hit_to_miss_flag=False, rtt_flag=True)

    regression(hit_to_miss_flag=True, rtt_flag=False)
    regression(hit_to_miss_flag=False, rtt_flag=False)



main()
