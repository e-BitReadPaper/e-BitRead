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
    plt.plot(x, y, linestyle='-', linewidth=3, label=label,markersize=15,markerfacecolor='none',marker=mol,
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
        colors = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]

            rects = plt.bar(index + bar_width*(polI - 2.5), 1.0*bitrateL[polI, :]/np.sum(bitrateL[polI, :])*100, bar_width,
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
        qoeL = [[], [], [], [], []]
        qoeAL = [[], [], [], [],[]]
        bitrateL = [[], [], [], [], []]
        markerOfLine = ['|', 'x', '^', 'o', 's']
        colorOfLine = ["#9c6ce8",color_dic['21'], "#efa446", "#ec7c61", "#6ccba9"]
        #colorOfLine = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        markerEdgeColor = ["#9c6ce8",color_dic['21'], "#efa446", "#ec7c61", "#6ccba9"]
        #markerEdgeColor = [color_dic['21'], "#ec7c61", "#efa446", "#9c6ce8", "#6ccba9"]
        for i in range(len(bitrateL)):
            bitrateL[i] = [0] * 5
        rebL = [[], [], [], [], []]
        bVarL = [[], [], [], [], []]
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
            # 遍历指定目录下每个trace文件
            for traceFName in os.listdir(dir + "/" + pol):
                # 打开trace文件,读取内容
                file = open(dir + "/" + pol + "/" + traceFName)
                # 跳过第一行表头和第二行初始值,读取所有数据行
                lines = file.readlines()[2:]
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
                    #bitrate = int(elements[7]) * 1.0 / 1024 / 1024  # Mbps
                    bitrate = int(elements[7]) * 1.0 / 1024   # Mbps
                    total_b += bitrate
                    total_osc += abs(last_br - bitrate)
                    bVarL[polI].append(abs(last_br - bitrate))
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

        # Average QoE bar chart
        title = 'qoe_' + str(cachePol)
        fig1 = plt.figure(title)
        fontSize = 20
        labels = ['HIGHEST', 'LOWER', 'CLOSEST', 'ME-BitCon', 'UTILITY']
        hatchIndex = [".","+","\\","o","-"]


        # qoe----------------------------
        ax = plt.subplot(1, 1, 1)
        bar_width = 0.4
        index = np.arange(polCount)
        bars = plt.bar(index, np.array(qoeAL), bar_width, edgecolor='w',color=["#9c6ce8",color_dic['21'], "#efa446", "#ec7c61", "#6ccba9"],
                zorder=2)
        patterns = ('.','+','\\','o','-')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)
        plt.ylabel('Average QoE', fontsize=30)
        ticksList = []
        for i in range(polCount):
            ticksList.append(labels[i])
        plt.xticks(index, ticksList, fontsize=25, rotation=-20)
        plt.yticks(fontsize=25)
        plt.ylim(0,0.9)
        plt.axhline(y=0, color='black', linestyle='-')
        #if cachePol == 0:
        #    plt.ylim(0.3,)

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.182, right=0.95, bottom=0.15, top=0.95)
        fig1.set_size_inches(8, 6, forward=True)
        plt.savefig(os.path.join(dir, "qoe" + str(cachePol) + ".pdf"))
        plt.show()

        # QoE CDF (创建新的figure)
        title = 'qoe_cdf_' + str(cachePol)
        fig_cdf = plt.figure(title)
        ax_cdf = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(qoeL[polI]), labels[polI], MOL, COL, MEC)

        plt.xlabel("QoE", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.legend(loc='lower right', handlelength=2, ncol=2, fontsize=18)
        ax_cdf.spines['bottom'].set_linewidth(1.5)
        ax_cdf.spines['left'].set_linewidth(1.5)
        ax_cdf.xaxis.set_tick_params(width=1.5)
        ax_cdf.yaxis.set_tick_params(width=1.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=25)
        # plt.xlim(0.8, 2.0)
        # plt.ylim(0.55, 1.01)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.99)
        fig_cdf.set_size_inches(7, 5, forward=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "qoe_cdf.pdf"))
        plt.show()

        # bitrate-----------------------
        bitrateL = np.array(bitrateL)
        title = 'bitrate_' + str(cachePol)
        fig2 = plt.figure(title)
        ax2 = plt.subplot(1, 1, 1)
        n_groups = 5
        opacity = 0.8
        bar_width = 0.16
        colors = ["#9c6ce8",color_dic['21'], "#efa446", "#ec7c61", "#6ccba9"]
        index = np.arange(n_groups)
        for polI in range(len(polDir)):
            pol = polDir[polI]

            rects = plt.bar(index + bar_width * (polI - 2.5),
                            1.0 * bitrateL[polI, :] / np.sum(bitrateL[polI, :]) * 100, bar_width,edgecolor='w',hatch=hatchIndex[polI],
                            alpha=opacity, color=colors[polI],
                            label=labels[polI], zorder=2)
        plt.xlabel("Bitrate(Kbps)", fontsize=30)
        plt.ylabel("Request Count(%)", fontsize=30)
        # plt.ylim(0, 55)
        plt.legend(loc='best', fontsize=20)
        plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid(axis='y', linestyle=':', color=color_dic['gray'], zorder=1)  # "#E0EEE0"
        plt.subplots_adjust(left=0.16, right=0.99, bottom=0.19, top=0.99)
        fig2.set_size_inches(7, 5, forward=True)
        plt.savefig(os.path.join(dir, "bitrate3.pdf"))
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
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Rebuffering Time(s)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.01)
        # plt.ylim(0.958, 1.0)''

        #################################
        ymajorFormatter = FormatStrFormatter('%1.3f')
        ymajorLocator = MultipleLocator(0.1)
        ax3.yaxis.set_major_formatter(ymajorFormatter)
        # ax3.yaxis.set_major_locator(ymajorLocator)
        #################################

        plt.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.97)
        fig3.set_size_inches(7, 5, forward=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "rebuffering3.pdf"))
        fig3.show()

        # bitrate variation-----------------
        labels2 = ['UTILITY', 'CLOSEST', 'LOWER','HIGHEST',  'ME-BitCon']
        title = 'bitratevariation_' + str(cachePol)
        fig4 = plt.figure(title)
        ax4 = plt.subplot(1, 1, 1)
        for polI in range(len(polDir)):
            pol = polDir[polI]
            MOL = markerOfLine[polI]
            COL = colorOfLine[polI]
            MEC = markerEdgeColor[polI]
            drawCDF(np.array(bVarL[polI]), labels2[polI], MOL, COL, MEC)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.xaxis.set_tick_params(width=1.5)
        ax4.yaxis.set_tick_params(width=1.5)
        #plt.legend(loc='best', handlelength=2, ncol=2, fontsize=18)
        plt.legend(loc='lower right', handlelength=2, ncol=1, fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel("Bitrate Variation(Mbps)", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
        plt.grid(linestyle=':', color=color_dic['gray'], zorder=1)
        # plt.xlim(-0.05, 3.0)
        # plt.ylim(0.82, 1.0)
        plt.subplots_adjust(left=0.182, right=0.95, bottom=0.15, top=0.95)
        fig4.set_size_inches(8, 6, forward=True)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "bitratevariation3.pdf"))
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

def main():
    #dir = "../data/第二个实验-0320/trace/"
    dir = "../data/RL_model/2024-12-19_14-30-50/test_trace_1/mine_test"
    # cdf_1(dir)
    # cdf_2(dir)
    cdf_3(dir)
    plt.show()
    # drawClient()

main()
