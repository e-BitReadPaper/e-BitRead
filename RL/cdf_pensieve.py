# coding:utf-8 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.api as sm  # recommended import according to the docs
import math
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib
import pandas as pd
from color import color_dic
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

def drawCDF(arr, label,mol,coline,mec):
    ecdf = sm.distributions.ECDF(arr)
    x = np.linspace(min(arr), max(arr))
    y = ecdf(x)
    plt.plot(x, y, linestyle='-', linewidth=3, label=label,markersize=20,markerfacecolor='none',marker=mol,
             color=coline,markeredgecolor=mec,markeredgewidth=3,markevery=5,zorder=2)#,fontsize=25)


def cdf_1(dir):
    for cachePol in [0]:
        qoeL = [[], [], [], [], []]
        bitrateL = [[], [], [], [], []]
        markerOfLine = ['x', 'o', '^', '|', 's']
        colorOfLine = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        markerEdgeColor = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        for i in range(len(bitrateL)):
            bitrateL[i] = [0] * 5
        rebL = [[], [], [], [], []]
        bVarL = [[], [], [], [], []]
        hitRL = [0]*5
        polDir = os.listdir(dir+"/trace"+str(cachePol))
        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            segCount = 0
            total_rew = 0
            total_reb = 0.0
            total_osc = 0.0
            total_b = 0.0
            for traceFName in os.listdir(dir+"/trace"+str(cachePol) + "/" + pol):
                file = open(dir+"/trace"+str(cachePol) + "/" + pol + "/" +traceFName)
                lines = file.readlines()[1:]
                segCount += len(lines)
                last_br = 0
                for line in lines:
                    # No chunkS Hit buffer bI aBI lastBI bitrate throughput hThroughput mThroughput downloadT rebufferT vT qoe reward time busy
                    elements = line.split("\t")
                    # hit -------------------------------------
                    hit = int(elements[2])
                    if hit == 1:
                        hitCount += 1
                    # bitrate index ---------------------------
                    bI = int(elements[5])
                    bitrateL[polI][bI] += 1
                    # bitrate variation -----------------------
                    bitrate = int(elements[7])*1.0 / 1024/1024 # Mbps
                    total_b += bitrate
                    total_osc += abs(last_br - bitrate)
                    bVarL[polI].append(abs(last_br - bitrate))
                    last_br = bitrate
                    # rebufferT -------------------------------
                    reb = float(elements[12])
                    total_reb += reb
                    if reb > 5:
                        reb = 5
                    rebL[polI].append(reb)
                    # qoe -------------------------------------
                    qoe = float(elements[14])
                    total_rew += qoe
                    if qoe < 0.8:
                        qoe = 0.8
                    qoeL[polI].append(qoe)
            hitRL[polI] = 1.0 * hitCount/segCount
            #print(cachePol, pol, "avg qoe=", total_rew/segCount, ",avg rebuf=", total_reb/segCount, ",
            #hit ratio=", 1.0 * hitCount/segCount, ",avg bitrate variation=", total_osc/segCount, ",avg bitrate=", total_b/segCount)

        title = 'qoe_'+str(cachePol)
        fig1 = plt.figure(title)
        fontSize = 20
        # qoe-----------------------
        ax1 = plt.subplot(1,1,1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(qoeL[polI]), pol,MOL,COL,MEC)
        plt.xlabel("QoE", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.legend(loc='best', fontsize=20) #loc='upper right',numpoints=1,handlelength=3,fontsize=20
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.xaxis.set_tick_params(width=1.5)
        ax1.yaxis.set_tick_params(width=1.5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        #plt.xlim(0.8, 2.0)
        #plt.ylim(0.55, 1.01)
        plt.grid(linestyle=':',color=color_dic['gray'],zorder=1)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.99)
        fig1.set_size_inches(7, 5, forward=True)
        fig1.show()



        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        title = 'bitrate_' + str(cachePol)
        fig2 = plt.figure(title)
        ax2 = plt.subplot(1,1,1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.16
        colors = ["#9c6ce8", color_dic['21'], "#efa446", "#ec7c61", "#6ccba9", "#7f7f7f"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            rects = plt.bar(index + bar_width * (polI - 2.5),
                          1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100, bar_width,
                          alpha=opacity, color=colors[polI],
                          label=pol, zorder=2)
        plt.xlabel("Bitrate(kbps)", fontsize=30)
        plt.ylabel("Request Count(%)", fontsize=30)
        #plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=18)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid(axis='y',linestyle=':',color=color_dic['gray'],zorder=1)    #"#E0EEE0"
        plt.subplots_adjust(left=0.145, right=0.99, bottom=0.18, top=0.99)
        fig2.set_size_inches(7, 5, forward=True)
        fig2.show()



        # rebuffer-----------------------
        title = 'rebuffering_' + str(cachePol)
        fig3 = plt.figure(title)
        ax3 = plt.subplot(1,1,1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(rebL[polI]), pol,MOL,COL,MEC)

        ax3.spines['bottom'].set_linewidth(1.5)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.xaxis.set_tick_params(width=1.5)
        ax3.yaxis.set_tick_params(width=1.5)
        plt.legend(loc='best',handlelength=2,ncol=2,fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Rebuffer Time(s)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':',color=color_dic['gray'],zorder=1)
        #plt.xlim(-0.05, 3.01)
        #plt.ylim(0.98, 1.0)

        #################################
        #################################

        ymajorFormatter = FormatStrFormatter('%1.3f')
        ymajorLocator = MultipleLocator(0.1)
        ax3.yaxis.set_major_formatter(ymajorFormatter)
        #ax3.yaxis.set_major_locator(ymajorLocator)

        #################################


        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig3.set_size_inches(7, 5, forward=True)
        fig3.show()



        # bitrate variation-----------------
        title = 'bitratevariation_' + str(cachePol)
        fig4 = plt.figure(title)
        ax4 = plt.subplot(1,1,1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(bVarL[polI]), pol,MOL,COL,MEC)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.xaxis.set_tick_params(width=1.5)
        ax4.yaxis.set_tick_params(width=1.5)
        plt.legend(loc='best',handlelength=2,ncol=2,fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Bitrate Variation(Mbps)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':',color=color_dic['gray'],zorder=1)
        #plt.xlim(-0.05, 3.0)
        #plt.ylim(0.82, 1.0)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig4.set_size_inches(7, 5, forward=True)
        fig4.show()




        # hit ratio----------------
        title = 'hitratio_' + str(cachePol)
        fig5 = plt.figure(title)
        ax5 = plt.subplot(1,1,1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.3
        index = np.arange(n_groups)
        plt.bar(index, np.array(hitRL)*100, bar_width,color=[color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"],zorder=2)
        plt.ylabel('Hit Ratio(%)',fontsize=30)
        #plt.xticks(index, ('Highest', 'No Plan', 'Closest', 'RELEASER', 'Lower'),fontsize=18)
        plt.xticks(index, (polDir[0], polDir[1], polDir[2], polDir[3], polDir[4]),fontsize=18)
        plt.yticks(fontsize=25)
        ax5.spines['bottom'].set_linewidth(1.5)
        ax5.spines['left'].set_linewidth(1.5)
        ax5.xaxis.set_tick_params(width=1.5)
        ax5.yaxis.set_tick_params(width=1.5)
        #plt.xlim(-0.05, 3.0)
        #plt.ylim(0, 75)
        plt.grid(axis='y',linestyle=':',color=color_dic['gray'],zorder=1)
        plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.99)
        fig5.set_size_inches(7, 5, forward=True)
        fig5.show()

def cdf_2(dir):
    for cachePol in [4]:
        qoeL = [[], [], [], [], []]
        bitrateL = [[], [], [], [], []]
        markerOfLine = ['x', 'o', '^', '|', 's']
        colorOfLine = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        markerEdgeColor = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        for i in range(len(bitrateL)):
            bitrateL[i] = [0] * 5
        rebL = [[], [], [], [], []]
        bVarL = [[], [], [], [], []]
        hitRL = [0] * 5
        polDir = os.listdir(dir + "/trace" + str(cachePol))
        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            segCount = 0
            total_rew = 0
            total_reb = 0.0
            total_osc = 0.0
            total_b = 0.0
            for traceFName in os.listdir(dir + "/trace" + str(cachePol) + "/" + pol):
                file = open(dir + "/trace" + str(cachePol) + "/" + pol + "/" + traceFName)
                lines = file.readlines()[1:]
                segCount += len(lines)
                last_br = 0
                for line in lines:
                    # No chunkS Hit buffer bI aBI lastBI bitrate throughput hThroughput mThroughput downloadT rebufferT vT qoe reward time busy
                    elements = line.split("\t")
                    # hit -------------------------------------
                    hit = int(elements[2])
                    if hit == 1:
                        hitCount += 1
                    # bitrate index ---------------------------
                    bI = int(elements[5])
                    bitrateL[polI][bI] += 1
                    # bitrate variation -----------------------
                    bitrate = int(elements[7]) * 1.0 / 1024 / 1024  # Mbps
                    total_b += bitrate
                    total_osc += abs(last_br - bitrate)
                    bVarL[polI].append(abs(last_br - bitrate))
                    last_br = bitrate
                    # rebufferT -------------------------------
                    reb = float(elements[12])
                    total_reb += reb
                    if reb > 5:
                        reb = 5
                    rebL[polI].append(reb)
                    # qoe -------------------------------------
                    qoe = float(elements[14])
                    total_rew += qoe
                    if qoe < 0.8:
                        qoe = 0.8
                    qoeL[polI].append(qoe)
            hitRL[polI] = 1.0 * hitCount / segCount
            # print(cachePol, pol, "avg qoe=", total_rew/segCount, ",avg rebuf=", total_reb/segCount, ",
            # hit ratio=", 1.0 * hitCount/segCount, ",avg bitrate variation=", total_osc/segCount, ",avg bitrate=", total_b/segCount)

        title = 'qoe_' + str(cachePol)
        fig1 = plt.figure(title)
        fontSize = 20
        # qoe-----------------------
        ax1 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(qoeL[polI]), pol, MOL, COL, MEC)
        plt.xlabel("QoE", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.legend(loc='best', fontsize=20)  # loc='upper right',numpoints=1,handlelength=3,fontsize=20
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.xaxis.set_tick_params(width=1.5)
        ax1.yaxis.set_tick_params(width=1.5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        # plt.xlim(0.8, 2.0)
        # plt.ylim(0.55, 1.01)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.99)
        fig1.set_size_inches(7, 5, forward=True)
        fig1.show()

        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        title = 'bitrate_' + str(cachePol)
        fig2 = plt.figure(title)
        ax2 = plt.subplot(1, 1, 1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.16
        colors = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]

            rects = plt.bar(index + bar_width * (polI - 2.5),
                            1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100, bar_width,
                            alpha=opacity, color=colors[polI],
                            label=pol, zorder=2)
        plt.xlabel("Bitrate(kbps)", fontsize=30)
        plt.ylabel("Request Count(%)", fontsize=30)
        # plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=18)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)  # "#E0EEE0"
        plt.subplots_adjust(left=0.145, right=0.99, bottom=0.18, top=0.99)
        fig2.set_size_inches(7, 5, forward=True)
        fig2.show()

        # rebuffer-----------------------
        title = 'rebuffering_' + str(cachePol)
        fig3 = plt.figure(title)
        ax3 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(rebL[polI]), pol, MOL, COL, MEC)

        ax3.spines['bottom'].set_linewidth(1.5)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.xaxis.set_tick_params(width=1.5)
        ax3.yaxis.set_tick_params(width=1.5)
        #plt.legend(loc='best', handlelength=2, ncol=2, fontsize=20)
        plt.legend(loc='lower right', handlelength=2, ncol=2, fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Rebuffer Time(s)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.01)
        # plt.ylim(0.958, 1.0)

        #################################
        ymajorFormatter = FormatStrFormatter('%1.3f')
        ymajorLocator = MultipleLocator(0.1)
        ax3.yaxis.set_major_formatter(ymajorFormatter)
        # ax3.yaxis.set_major_locator(ymajorLocator)
        #################################

        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig3.set_size_inches(7, 5, forward=True)
        fig3.show()

        # bitrate variation-----------------
        title = 'bitratevariation_' + str(cachePol)
        fig4 = plt.figure(title)
        ax4 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(bVarL[polI]), pol, MOL, COL, MEC)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.xaxis.set_tick_params(width=1.5)
        ax4.yaxis.set_tick_params(width=1.5)
        plt.legend(loc='best', handlelength=2, ncol=2, fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Bitrate Variation(Mbps)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.0)
        # plt.ylim(0.82, 1.0)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig4.set_size_inches(7, 5, forward=True)
        fig4.show()

        # hit ratio----------------
        title = 'hitratio_' + str(cachePol)
        fig5 = plt.figure(title)
        ax5 = plt.subplot(1, 1, 1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.3
        index = np.arange(n_groups)
        plt.bar(index, np.array(hitRL) * 100, bar_width,
                color=[color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"], zorder=2)
        plt.ylabel('Hit Ratio(%)', fontsize=30)
        #plt.xticks(index, ('Highest', 'No Plan', 'Closest', 'RELEASER', 'Lower'), fontsize=18)
        plt.xticks(index, (polDir[0], polDir[1], polDir[2], polDir[3], polDir[4]), fontsize=18)
        plt.yticks(fontsize=25)
        ax5.spines['bottom'].set_linewidth(1.5)
        ax5.spines['left'].set_linewidth(1.5)
        ax5.xaxis.set_tick_params(width=1.5)
        ax5.yaxis.set_tick_params(width=1.5)
        # plt.xlim(-0.05, 3.0)
        # plt.ylim(0, 75)
        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.99)
        fig5.set_size_inches(7, 5, forward=True)
        fig5.show()

def cdf_3(dir):
    for cachePol in [1]:
        # 先获取实际的策略数量
        polDir = os.listdir(dir)
        polCount = len(polDir)
        print(f"策略数量: {polCount}")
        print(f"策略列表: {polDir}")

         # 打印策略和标签的对应关系
        labels = ['HIGHEST', 'LOWER', 'CLOSEST', 'ME-BitCon', 'UTILITY']
        print("策略对应关系：")
        for i in range(len(polDir)):
            print(f"目录名: {polDir[i]} -> 标签: {labels[i]}")



        qoeL = [[], [], [], [], []]
        qoeAL = [[], [], [], [], []]
        bitrateL = [[], [], [], [], []]
        markerOfLine = ['|', 'x',  'o', 's', '*']
        colorOfLine = ["#9c6ce8", color_dic['21'],  "#ec7c61", "#6ccba9", "#7f7f7f"]
        #colorOfLine = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        markerEdgeColor = ["#9c6ce8", color_dic['21'],  "#ec7c61", "#6ccba9", "#7f7f7f"]
        #markerEdgeColor = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        # qoeL = [[] for _ in range(6)]  # 改为6个策略
        # qoeAL = [[] for _ in range(6)]
        # bitrateL = [[] for _ in range(6)]
        # markerOfLine = ['|', 'x', '^', 'o', 's', '*']  # 增加一个标记
        # colorOfLine = ["#9c6ce8", color_dic['21'], "#efa446", "#ec7c61", "#6ccba9", "#7f7f7f"]  # 增加一个颜色
        # markerEdgeColor = ["#9c6ce8", color_dic['21'], "#efa446", "#ec7c61", "#6ccba9", "#7f7f7f"]


        for i in range(len(bitrateL)):
            bitrateL[i] = [0] * 5
            
        rebL = [[] for _ in range(5)]
        bVarL = [[] for _ in range(5)]
        hitRL = [0] * 5
        # polDir = os.listdir(dir + "/trace" + str(cachePol))
        polDir = os.listdir(dir)
        polCount = len(polDir)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            segCount = 0
            total_rew = 0
            total_reb = 0.0
            total_osc = 0.0
            total_b = 0.0
            # 添加对client_0到client_9的循环
            for client_id in range(10):
                client_dir = f"client_{client_id}"
                client_path = os.path.join(dir, pol, client_dir)
                
                # 检查client目录是否存在
                if not os.path.exists(client_path):
                    continue
                    
                for traceFName in os.listdir(client_path):
                    # 打开trace文件,读取内容
                    file = open(os.path.join(client_path, traceFName))
                    # 跳过第一行表头和第二行初始值,读取所有数据行
                    lines = file.readlines()[2:]
                    segCount += len(lines)
                    last_br = 0
                    for line in lines:
                        # No chunkS Hit buffer bI aBI lastBI bitrate throughput hThroughput mThroughput downloadT rebufferT vT qoe reward time busy
                        elements = line.split("\t")
                        # hit -------------------------------------
                        hit = elements[2].strip()
                        if hit == 'True':
                            hit = 1
                        elif hit == 'False':
                            hit = 0
                        else:
                            hit = int(hit)
                            
                        if hit == 1:
                            hitCount += 1
                        # bitrate index ---------------------------
                        bI = int(elements[5])
                        bitrateL[polI][bI] += 1
                        # bitrate variation -----------------------
                        #bitrate = int(elements[7]) * 1.0 / 1024 / 1024  # Mbps
                        bitrate = int(elements[7]) * 1.0 / 1024   # Mbps
                        total_b += bitrate
                        total_osc += abs(last_br - bitrate)
                        # 计算码率变化
                        bitrate_var = abs(last_br - bitrate)
                        
                        # 对ME-BitCon策略进行特殊处理
                        if labels[polI] == 'ME-BitCon':
                            # 只记录码率变化小于阈值的数据
                            #if bitrate_var <= 0.35:  # 可以调整阈值
                            bVarL[polI].append(bitrate_var)
                        else:
                            # 其他策略正常记录所有数据
                            bVarL[polI].append(bitrate_var)
                        last_br = bitrate
                        # rebufferT -------------------------------
                        #reb = float(elements[-4]) 原来这里是倒数第四个元素，但是倒数第四个元素应是qoe
                        reb = float(elements[-5])
                        total_reb += reb
                        if reb > 5:
                            reb = 5
                        rebL[polI].append(reb)
                        # qoe -------------------------------------
                        qoe = float(elements[-3])
                        total_rew += qoe
                        #if qoe < 0.8:
                        #    qoe = 0.8
                        qoeL[polI].append(qoe)

                    hitRL[polI] = 1.0 * hitCount / segCount
                    qoeAL[polI] = total_rew/segCount
                # print(cachePol, pol, "avg qoe=", total_rew/segCount, ",avg rebuf=", total_reb/segCount, ",
                # hit ratio=", 1.0 * hitCount/segCount, ",avg bitrate variation=", total_osc/segCount, ",avg bitrate=", total_b/segCount)

        title = 'qoe_' + str(cachePol)
        fig1 = plt.figure(title)
        fontSize = 20
        labels = ['HIGHEST', 'LOWER', 'CLOSEST', 'ME-BitCon', 'UTILITY']
        hatchIndex = [".","+","o","-","*"]  # 减少为5种图案




        # qoe----------------------------
        ax = plt.subplot(1, 1, 1)
        bar_width = 0.4
        index = np.arange(polCount)
        bars = plt.bar(index, np.array(qoeAL), bar_width, edgecolor='w',
                color=["#9c6ce8", color_dic['21'], "#ec7c61", "#6ccba9", "#7f7f7f"],  # 增加一个颜色
                zorder=2)
        patterns = ('.','+','o','-','*')  
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)
        plt.ylabel('Average QoE', fontsize=28)
        ticksList = []
        for i in range(polCount):
            ticksList.append(labels[i])
        plt.xticks(index, ticksList, fontsize=25, rotation=-20)
        plt.yticks(fontsize=28)
        plt.ylim(0,1.2)
        plt.axhline(y=0, color='black', linestyle='-')
        #if cachePol == 0:
        #    plt.ylim(0.3,)

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        # 设置y轴刻度为两位小数
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.182, right=0.95, bottom=0.20, top=0.95)
        fig1.set_size_inches(8, 6, forward=True)
        plt.savefig(os.path.join(dir, "qoe" + str(cachePol) + ".pdf"))
        plt.show()



        # qoe cdf-----------------------
        '''
        ax1 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(qoeL[polI]), pol, MOL, COL, MEC)
        plt.xlabel("QoE", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.legend(loc='best', fontsize=20)  # loc='upper right',numpoints=1,handlelength=3,fontsize=20
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.xaxis.set_tick_params(width=1.5)
        ax1.yaxis.set_tick_params(width=1.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=25)
        # plt.xlim(0.8, 2.0)
        # plt.ylim(0.55, 1.01)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.99)
        fig1.set_size_inches(7, 5, forward=True)
        fig1.show()'''


        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        title = 'bitrate_' + str(cachePol)
        fig2 = plt.figure(title)
        ax2 = plt.subplot(1, 1, 1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.16
        colors = ["#9c6ce8", color_dic['21'], "#ec7c61", "#6ccba9", "#7f7f7f"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            rects = plt.bar(index + bar_width * (polI - 2.5),
                            1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100, bar_width,edgecolor='w',hatch=hatchIndex[polI],
                            alpha=opacity, color=colors[polI],
                            label=labels[polI], zorder=2)
        plt.xlabel("Bitrate(Kbps)", fontsize=28)
        plt.ylabel("Request Count(%)", fontsize=28)
        # plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=18)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)  # "#E0EEE0"
        plt.subplots_adjust(left=0.16, right=0.99, bottom=0.19, top=0.99)
        fig2.set_size_inches(7, 5, forward=True)
        plt.savefig(os.path.join(dir, "bitrate_RL.pdf"))
        fig2.show()

        # rebuffer-----------------------
        title = 'rebuffering_' + str(cachePol)
        fig3 = plt.figure(title)
        ax3 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(rebL[polI]), labels[polI], MOL, COL, MEC)

        ax3.spines['bottom'].set_linewidth(1.5)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.xaxis.set_tick_params(width=1.5)
        ax3.yaxis.set_tick_params(width=1.5)
        #plt.legend(loc='best', handlelength=2, ncol=2, fontsize=20)
        plt.legend(loc='lower right', handlelength=2, ncol=2, fontsize=18)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("Rebuffering Time(s)", fontsize=28)
        plt.ylabel("CDF", fontsize=28)
        # 设置y轴刻度为两位小数
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.yaxis.set_major_locator(MultipleLocator(0.04))  # 将刻度间距设为0.05
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.01)
        # plt.ylim(0.958, 1.0)''

        #################################
        # ymajorFormatter = FormatStrFormatter('%1.3f')
        # ymajorLocator = MultipleLocator(0.1)
        # ax3.yaxis.set_major_formatter(ymajorFormatter)
        # ax3.yaxis.set_major_locator(ymajorLocator)
        #################################

        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig3.set_size_inches(7, 5, forward=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "rebuffering_RL.pdf"))
        fig3.show()

        # bitrate variation-----------------
        title = 'bitratevariation_' + str(cachePol)
        fig4 = plt.figure(title)
        ax4 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(bVarL[polI]), labels[polI], MOL, COL, MEC)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.xaxis.set_tick_params(width=1.5)
        ax4.yaxis.set_tick_params(width=1.5)
        # 设置y轴刻度为两位小数并调整间距
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax4.yaxis.set_major_locator(MultipleLocator(0.04))  # 将刻度间距设为0.10
        #plt.legend(loc='best', handlelength=2, ncol=2, fontsize=18)
        plt.legend(loc='lower right', handlelength=2, ncol=1, fontsize=20)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("Bitrate Variation(Mbps)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.0)
        # plt.ylim(0.82, 1.0)
        plt.subplots_adjust(left=0.182, right=0.95, bottom=0.16, top=0.95)
        fig4.set_size_inches(8, 6, forward=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "bitratevariation_RL.pdf"))
        fig4.show()

        # hit ratio----------------
        '''title = 'hitratio_' + str(cachePol)
        fig5 = plt.figure(title)
        ax5 = plt.subplot(1, 1, 1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.3
        index = np.arange(n_groups)
        plt.bar(index, np.array(hitRL) * 100, bar_width,
                color=[color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"], zorder=2)
        plt.ylabel('Hit Ratio(%)', fontsize=30)
        #plt.xticks(index, ('Highest', 'No Plan', 'Closest', 'RELEASER', 'Lower'), fontsize=18)
        plt.xticks(index, (labels[0], labels[1], labels[2], labels[3], labels[4]), fontsize=18)
        plt.yticks(fontsize=25)
        ax5.spines['bottom'].set_linewidth(1.5)
        ax5.spines['left'].set_linewidth(1.5)
        ax5.xaxis.set_tick_params(width=1.5)
        ax5.yaxis.set_tick_params(width=1.5)
        # plt.xlim(-0.05, 3.0)
        # plt.ylim(0, 75)
        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.99)
        fig5.set_size_inches(7, 5, forward=True)
        fig5.show()'''

        # 创建字典来存储各项指标的总和
        metrics = {
            'qoe': [[] for _ in range(5)],
            'bitrate': [[] for _ in range(5)],
            'rebuffer': [[] for _ in range(5)],
            'bitrate_var': [[] for _ in range(5)]
        }
        
        for polI in range(len(polDir)):
            pol = polDir[polI]
            hitCount = 0
            segCount = 0
            
            for client_id in range(10):
                client_dir = f"client_{client_id}"
                client_path = os.path.join(dir, pol, client_dir)
                
                # 检查client目录是否存在
                if not os.path.exists(client_path):
                    continue
                    
                for traceFName in os.listdir(client_path):
                    file = open(os.path.join(client_path, traceFName))
                    lines = file.readlines()[2:]
                    segCount += len(lines)
                    last_br = 0
                    for line in lines:
                        elements = line.split("\t")
                        
                        # 收集各项指标
                        bitrate = int(elements[7]) * 1.0 / 1024  # Mbps
                        metrics['bitrate'][polI].append(bitrate)
                        
                        bitrate_var = abs(last_br - bitrate)
                        metrics['bitrate_var'][polI].append(bitrate_var)
                        
                        reb = float(elements[-5])
                        metrics['rebuffer'][polI].append(reb)
                        
                        qoe = float(elements[-3])
                        metrics['qoe'][polI].append(qoe)
                        
                        last_br = bitrate

        # 打印各项指标的均值
        print("\n各策略指标均值统计:")
        print("=" * 50)
        
        for i in range(len(labels)):
            print(f"\n{labels[i]}策略:")
            print(f"平均 QoE: {np.mean(metrics['qoe'][i]):.4f}")
            print(f"平均码率: {np.mean(metrics['bitrate'][i]):.4f} Mbps")
            print(f"平均重缓冲时间: {np.mean(metrics['rebuffer'][i]):.4f} s")
            print(f"平均码率变化: {np.mean(metrics['bitrate_var'][i]):.4f} Mbps")
            print("-" * 30)

    # 创建第一个组合图（Bitrate和QoE）
    fig1 = plt.figure(figsize=(24, 8))
    
    # 创建带有顶部空间的子图网格
    gs = fig1.add_gridspec(2, 2, height_ratios=[0.2, 1])
    
    # Bitrate图（左下）
    ax1_left = fig1.add_subplot(gs[1, 0])
    n_groups = 5
    opacity = 0.8
    bar_width = 0.16
    colors = ["#9c6ce8", color_dic['21'], "#ec7c61", "#6ccba9", "#7f7f7f"]
    index = np.arange(n_groups) * 1.1
    for polI in range(len(polDir)):
        pol = polDir[polI]
        rects = plt.bar(index + bar_width * (polI - 2.5),
                      1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100, 
                      bar_width, edgecolor='w', hatch=hatchIndex[polI],
                      alpha=opacity, color=colors[polI],
                      label=labels[polI], zorder=2)
    
    plt.xlabel("Bitrate(Kbps)", fontsize=40)
    plt.ylabel("Request Count(%)", fontsize=40)
    plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # QoE图（右下）
    ax1_right = fig1.add_subplot(gs[1, 1])
    bar_width = 0.4
    index = np.arange(len(polDir))
    bars = plt.bar(index, np.array(qoeAL), bar_width, edgecolor='w',
                  color=["#9c6ce8", color_dic['21'], "#ec7c61", "#6ccba9", "#7f7f7f"],
                  zorder=2)
    
    patterns = ('.','+','o','-','*')
    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)
    #plt.xlabel("Schemes", fontsize=40, labelpad=15)
    plt.ylabel('Average QoE', fontsize=40)
    # 添加倾斜的策略名称标签
    plt.xticks(index, labels, rotation=-25, fontsize=40)  # 使用labels变量，-25度倾斜
    plt.yticks(fontsize=40)
    plt.ylim(0,1.2)
    plt.axhline(y=0, color='black', linestyle='-')
    
    ax1_right.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # 添加共同的legend
    legend = fig1.legend(bbox_to_anchor=(0.5, 0.91),
                       loc='lower center',
                       ncol=3,
                       fontsize=40,
                       handlelength=2)
    
    plt.subplots_adjust(left=0.08, right=0.92, 
                       bottom=0.24, top=1.03, 
                       wspace=0.25, hspace=0)
    
    plt.savefig(os.path.join(dir, "bitrate_qoe_combined_RL.pdf"), 
               bbox_inches='tight',
               bbox_extra_artists=(legend,))
    
    # 创建第二个组合图（Bitrate Variation和Rebuffer）
    fig2 = plt.figure(figsize=(24, 10))
    
    # Bitrate Variation图（左）
    ax2_left = plt.subplot(1, 2, 1)
    for polI in range(len(polDir)):
        pol = polDir[polI]
        MOL = markerOfLine[polI]
        COL = colorOfLine[polI]
        MEC = markerEdgeColor[polI]
        drawCDF(np.array(bVarL[polI]), labels[polI], MOL, COL, MEC)
    
    plt.xlabel("Bitrate Variation(Mbps)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # Rebuffer图（右）
    ax2_right = plt.subplot(1, 2, 2)
    for polI in range(len(polDir)):
        pol = polDir[polI]
        MOL = markerOfLine[polI]
        COL = colorOfLine[polI]
        MEC = markerEdgeColor[polI]
        drawCDF(np.array(rebL[polI]), '_nolegend_', MOL, COL, MEC)
    
    plt.xlabel("Rebuffering Time(s)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # 添加共同的legend
    fig2.legend(bbox_to_anchor=(0.5, 0.92),
               loc='center',
               ncol=3,
               fontsize=39,
               handlelength=2)
    
    plt.subplots_adjust(left=0.09, right=0.99, 
                       bottom=0.12, top=0.82, 
                       wspace=0.25, hspace=0)
    plt.savefig(os.path.join(dir, "bitrate_var_rebuffer_combined_RL.pdf"))

def main():
    #dir = "../data/第二个实验-0320/trace/"
    dir = "../data/RL_model/2024-12-19_14-30-50/test_trace_multi_client_0/FCC_test_2"
    # cdf_1(dir)
    # cdf_2(dir)
    cdf_3(dir)
    plt.show()
    # drawClient()

main()
