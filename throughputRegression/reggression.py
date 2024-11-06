import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
from sklearn.externals import joblib
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import time
import os
from prettytable import PrettyTable

HIT_FLAG = True


def run(dt, rttFlag=True, method="xgboost", dir="../../data/throughputRelation/data/1"):

    train=pd.read_csv(dir + "/train/sample.csv", delimiter=' ')
    test=pd.read_csv(dir + "/test/sample.csv", delimiter=' ')
    test2=pd.read_csv(dir + "/test/sample.csv", delimiter=' ')
    len_train=train.shape[0]
    datas=pd.concat([train,test], sort=False)
    if HIT_FLAG == True:
        if rttFlag:
            train.drop(['hTime', 'maxMT', 'reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
            test.drop(['hTime', 'maxMT', 'reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
        else:
            train.drop(['hTime', 'rtt', 'maxMT', 'reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
            test.drop(['hTime', 'rtt', 'maxMT', 'reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
    else:
        if rttFlag:
            train.drop(['mTime', 'maxMT','reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
            test.drop(['mTime', 'maxMT','reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
        else:
            train.drop(['mTime','rtt', 'maxMT','reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)
            test.drop(['mTime', 'rtt', 'maxMT','reqCount', 'avgMT', 'totalMT'], axis=1, inplace=True)


    len_train=train.shape[0]

    datas=pd.concat([train,test], sort=False)
    #
    # skew_ = datas.select_dtypes(include=['int', 'float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # skew_df = pd.DataFrame({'Skew': skew_})
    # skewed_df = skew_df[(skew_df['Skew'] > 0.5) | (skew_df['Skew'] < -0.5)]
    #
    # print(skewed_df.index)
    if HIT_FLAG == True:
        if rttFlag:
            skew_column = ['mTime', 'size', 'rtt', 'mThroughput']
        else:
            skew_column = ['mTime', 'size', 'mThroughput']
    else:
        if rttFlag:
            skew_column = ['hTime', 'size', 'rtt', 'hThroughput']
        else:
            skew_column = ['hTime', 'size', 'hThroughput']


    lam=0.1
    for col in skew_column:
        train[col]=boxcox1p(train[col],lam)
        test[col]=boxcox1p(test[col],lam)
    if HIT_FLAG == True:
        train['hThroughput'] = np.log(train['hThroughput'])
        x = train.drop('hThroughput', axis=1)
        y = train['hThroughput']
        x_test = test.drop('hThroughput', axis=1)
    else:
        train['mThroughput'] = np.log(train['mThroughput'])
        x = train.drop('mThroughput', axis=1)
        y = train['mThroughput']
        x_test = test.drop('mThroughput', axis=1)

    print(x.columns)
    if method=="lasso":
        model = Lasso(max_iter=1e7, alpha =0.0001, random_state=1)
    elif method=="ridge":
        model = Ridge(alpha=14.5)
    elif method=="lgbm":
        model = LGBMRegressor(objective='regression',
                                       max_depth=6,
                                       num_leaves=4,
                                       learning_rate=0.05,
                                       n_estimators=5000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
    elif method == "LinearRegression":
        model = LinearRegression()
    else:
        # model = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:gamma')
        model = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=5000, silent=False, objective='reg:gamma')
    startTime = time.time()
    model.fit(x, y)
    endTime = time.time()
    trainTime= endTime - startTime
    if HIT_FLAG:
        modelDir = dir + "/m2h/model/" + dt
    else:
        modelDir = dir + "/h2m/model/" + dt
    if rttFlag:
        modelDir += "/rtt"
    else:
        modelDir += "/nortt"

    if os.path.exists(modelDir) == False:
        os.makedirs(modelDir)

    joblib.dump(model, modelDir + "/" + method + ".m")

    startTime= time.time()
    y_pred = model.predict(x_test)
    endTime= time.time()

    plt.rcParams['figure.figsize'] = (4.0, 4.0)
    fig, ax = plt.subplots()
    if HIT_FLAG:
        y_test = test2['hThroughput'] / 1024 / 1024
    else:
        y_test=test2['mThroughput']/1024/1024

    y_pred=np.exp(y_pred)/1024/1024
    # ax.scatter(y_test, y_pred, s=2, alpha=0.1)
    # ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'k--', lw=4)
    # ax.set_xlabel('Real throughput/Mbps')
    # ax.set_ylabel('Predicted throughput/Mbps')
    # plt.title(method)
    # plt.savefig("../plot/"+method+".pdf", bbox_inches = 'tight')
    # plt.show()

    MSE_loss = np.average(np.square(y_pred-y_test))
    print(method,"MSE_loss =",MSE_loss, "testTime=",endTime-startTime, "trainTime=",trainTime)
    return method ,MSE_loss, endTime-startTime ,trainTime

def main():
    global HIT_FLAG
    table = PrettyTable(["algorithm", "MSE_loss", "testTime", "trainTime"])
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
    # methods = ["lasso", "xgboost", "ridge", "lgbm", "LinearRegression"]
    # for m in methods:
    #     method, MSE_loss, testTime, trainTime = \
    #         run(method=m, dir="../../data/throughputRelation/data/2", dt=dt, rttFlag=True)
    #     table.add_row([method, MSE_loss, testTime, trainTime])
    # for m in methods:
    #     method, MSE_loss, testTime, trainTime = \
    #         run(method=m, dir="../../data/throughputRelation/data/2", dt=dt, rttFlag=False)
    #     table.add_row([method, MSE_loss, testTime, trainTime])
    # print(table.get_string())

    HIT_FLAG = True
    run(method="LinearRegression" ,dir="../../data/throughputRelation/data/2", dt=dt, rttFlag=True)
    HIT_FLAG = False
    run(method="LinearRegression" ,dir="../../data/throughputRelation/data/2", dt=dt, rttFlag=True)

main()
