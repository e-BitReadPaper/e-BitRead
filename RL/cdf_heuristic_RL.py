import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
from color import color_dic
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

def drawCDF(arr, label, mol, coline, mec):
    ecdf = sm.distributions.ECDF(arr)
    x = np.linspace(0, 3, 11)
    y = ecdf(x)
    
    plt.plot(x, y, linestyle='-', linewidth=3, 
            label=label, markersize=25, markerfacecolor='none',
            marker=mol, color=coline, markeredgecolor=mec,
            markeredgewidth=3, zorder=2)

def draw_combined_plots():
    # 数据路径
    base_dir = "../data/RL_model/2024-12-19_14-30-50/test_trace_multi_client_0"
    heuristic_dir = os.path.join(base_dir, "mine_test")
    ml_dir = os.path.join(base_dir, "FCC_test_2")
    
    # Heuristic策略配置（左图）
    heuristic_colors = ["#9c6ce8", color_dic['21'], "#efa446", "#ec7c61", "#6ccba9", "#7f7f7f"]
    heuristic_hatches = ['.', '+', '\\', 'o', '-', '*']
    heuristic_markers = ['|', 'x', '^', 'o', 's', '*']
    heuristic_labels = ['HIGHEST', 'LOWER', 'LFBM', 'CLOSEST', 'ME-BitCon', 'UTILITY']
    
    # ML策略配置（右图，扩展为6种策略）
    ml_colors = ["#9c6ce8", color_dic['21'], "#efa446", "#ec7c61", "#6ccba9", "#7f7f7f"]
    ml_hatches = ['.', '+', '\\', 'o', '-', '*']
    ml_markers = ['|', 'x', '^', 'o', 's', '*']
    ml_labels = ['HIGHEST', 'LOWER', 'LFBM', 'CLOSEST', 'ME-BitCon', 'UTILITY']
    
    # 处理数据...
    # [这里需要添加数据处理的代码，与原cdf_3函数相同]
    
    # 1. Bitrate分布图
    fig1, (ax1_left, ax1_right) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Heuristic Bitrate图（左）
    plt.sca(ax1_left)
    n_groups = 5
    opacity = 0.8
    bar_width = 0.16
    index = np.arange(n_groups) * 1.1
    
    for i, label in enumerate(heuristic_labels):
        plt.bar(index + bar_width * (i - 2.5),
               1.0 * heuristic_bitrate[i, :] / np.sum(heuristic_bitrate[i, :]) * 100,
               bar_width, edgecolor='w', hatch=heuristic_hatches[i],
               alpha=opacity, color=heuristic_colors[i],
               label=label, zorder=2)
    
    plt.xlabel("Bitrate(Kbps)", fontsize=40)
    plt.ylabel("Request Count(%)", fontsize=40)
    plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # ML Bitrate图（右）
    plt.sca(ax1_right)
    for i, label in enumerate(ml_labels):
        plt.bar(index + bar_width * (i - 2.5),
               1.0 * ml_bitrate[i, :] / np.sum(ml_bitrate[i, :]) * 100,
               bar_width, edgecolor='w', hatch=ml_hatches[i],
               alpha=opacity, color=ml_colors[i],
               label='_nolegend_', zorder=2)
    
    plt.xlabel("Bitrate(Kbps)", fontsize=40)
    plt.ylabel("Request Count(%)", fontsize=40)
    plt.xticks(index, ('350', '600', '1000', '2000', '3000'), fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # 添加共同的legend
    fig1.legend(bbox_to_anchor=(0.5, 1.05),
               loc='center',
               ncol=3,
               fontsize=40,
               handlelength=2)
    
    plt.subplots_adjust(left=0.08, right=0.92,
                       bottom=0.15, top=0.85,
                       wspace=0.25)
    
    plt.savefig("../data/data/draw_pic/data/bitrate_distribution_combined.pdf")
    
    # 2. QoE图
    fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Heuristic QoE图（左）
    plt.sca(ax2_left)
    bar_width = 0.4
    index = np.arange(len(heuristic_labels))
    bars = plt.bar(index, heuristic_qoe, bar_width, edgecolor='w',
                  color=heuristic_colors, zorder=2)
    
    for bar, pattern in zip(bars, heuristic_hatches):
        bar.set_hatch(pattern)
    
    plt.xlabel("Schemes", fontsize=40)
    plt.ylabel('Average QoE', fontsize=40)
    plt.xticks(index, heuristic_labels, rotation=-25, fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(0, 1.0)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # ML QoE图（右）
    plt.sca(ax2_right)
    bars = plt.bar(index, ml_qoe, bar_width, edgecolor='w',
                  color=ml_colors, zorder=2)
    
    for bar, pattern in zip(bars, ml_hatches):
        bar.set_hatch(pattern)
    
    plt.xlabel("Schemes", fontsize=40)
    plt.ylabel('Average QoE', fontsize=40)
    plt.xticks(index, ml_labels, rotation=-25, fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(0, 1.0)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    
    # 添加共同的legend
    fig2.legend(bbox_to_anchor=(0.5, 1.05),
               loc='center',
               ncol=3,
               fontsize=40,
               handlelength=2)
    
    plt.subplots_adjust(left=0.08, right=0.92,
                       bottom=0.15, top=0.85,
                       wspace=0.25)
    
    plt.savefig("../data/data/draw_pic/data/qoe_combined.pdf")
    
    # 3. Bitrate Variation图
    fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Heuristic Bitrate Variation图（左）
    plt.sca(ax3_left)
    for i, label in enumerate(heuristic_labels):
        drawCDF(np.array(heuristic_bvar[i]), label, heuristic_markers[i], heuristic_colors[i], None)
    
    plt.xlabel("Bitrate Variation(Mbps)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # ML Bitrate Variation图（右）
    plt.sca(ax3_right)
    for i, label in enumerate(ml_labels):
        drawCDF(np.array(ml_bvar[i]), '_nolegend_', ml_markers[i], ml_colors[i], None)
    
    plt.xlabel("Bitrate Variation(Mbps)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # 添加共同的legend
    fig3.legend(bbox_to_anchor=(0.5, 1.05),
               loc='center',
               ncol=3,
               fontsize=39,
               handlelength=2)
    
    plt.subplots_adjust(left=0.08, right=0.92,
                       bottom=0.15, top=0.85,
                       wspace=0.25)
    
    plt.savefig("../data/data/draw_pic/data/bitrate_variation_combined.pdf")
    
    # 4. Rebuffering Time图
    fig4, (ax4_left, ax4_right) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Heuristic Rebuffering Time图（左）
    plt.sca(ax4_left)
    for i, label in enumerate(heuristic_labels):
        drawCDF(np.array(heuristic_reb[i]), label, heuristic_markers[i], heuristic_colors[i], None)
    
    plt.xlabel("Rebuffering Time(s)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # ML Rebuffering Time图（右）
    plt.sca(ax4_right)
    for i, label in enumerate(ml_labels):
        drawCDF(np.array(ml_reb[i]), '_nolegend_', ml_markers[i], ml_colors[i], None)
    
    plt.xlabel("Rebuffering Time(s)", fontsize=39)
    plt.ylabel("CDF", fontsize=39)
    plt.grid(linestyle='-.', color='gray', zorder=1)
    plt.xticks(fontsize=39)
    plt.yticks(fontsize=39)
    
    # 添加共同的legend
    fig4.legend(bbox_to_anchor=(0.5, 1.05),
               loc='center',
               ncol=3,
               fontsize=39,
               handlelength=2)
    
    plt.subplots_adjust(left=0.08, right=0.92,
                       bottom=0.15, top=0.85,
                       wspace=0.25)
    
    plt.savefig("../data/data/draw_pic/data/rebuffering_time_combined.pdf")

if __name__ == "__main__":
    draw_combined_plots()
