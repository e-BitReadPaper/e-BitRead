import numpy as np  
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.use('TkAgg')

def create_membership_functions():  
    # 电池电量范围 (0-100%)  
    battery_x = np.arange(0, 101, 1)  
    
    # 电池电量隶属函数  
    battery_s = np.zeros_like(battery_x, dtype=float)  
    battery_m = np.zeros_like(battery_x, dtype=float)  
    battery_l = np.zeros_like(battery_x, dtype=float)  

    # Small (S)  
    battery_s[battery_x <= 30] = 1  
    mask = (battery_x > 30) & (battery_x < 50)  
    battery_s[mask] = 1 - (battery_x[mask] - 30) / 20  

    # Medium (M)  
    mask = (battery_x >= 30) & (battery_x <= 50)  
    battery_m[mask] = (battery_x[mask] - 30) / 20  
    mask = (battery_x >= 50) & (battery_x <= 70)  
    battery_m[mask] = 1 - (battery_x[mask] - 50) / 20  

    # Large (L)  
    battery_l[battery_x >= 70] = 1  
    mask = (battery_x > 50) & (battery_x < 70)  
    battery_l[mask] = (battery_x[mask] - 50) / 20  

    # 计算能力范围 (400-2000)  
    power_x = np.arange(400, 2001, 1)  
    
    # 计算能力隶属函数  
    power_s = np.zeros_like(power_x, dtype=float)  
    power_m = np.zeros_like(power_x, dtype=float)  
    power_l = np.zeros_like(power_x, dtype=float)  

    # Small (S)  
    power_s[power_x <= 800] = 1  
    mask = (power_x > 800) & (power_x < 1200)  
    power_s[mask] = 1 - (power_x[mask] - 800) / 400  

    # Medium (M)  
    mask = (power_x >= 800) & (power_x <= 1200)  
    power_m[mask] = (power_x[mask] - 800) / 400  
    mask = (power_x >= 1200) & (power_x <= 1600)  
    power_m[mask] = 1 - (power_x[mask] - 1200) / 400  

    # Large (L)  
    power_l[power_x >= 1600] = 1  
    mask = (power_x > 1200) & (power_x < 1600)  
    power_l[mask] = (power_x[mask] - 1200) / 400  

    return (battery_x, battery_s, battery_m, battery_l,  
            power_x, power_s, power_m, power_l)  

def plot_membership_functions():  
    (battery_x, battery_s, battery_m, battery_l,  
     power_x, power_s, power_m, power_l) = create_membership_functions()  
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  
    
    # 绘制电池电量隶属函数  
    ax1.plot(battery_x, battery_s, 'r-', label='Small (S)', linewidth=2)  
    ax1.plot(battery_x, battery_m, 'b-', label='Medium (M)', linewidth=2)  
    ax1.plot(battery_x, battery_l, 'g-', label='Large (L)', linewidth=2)  
    
    ax1.set_xlabel('Battery Level (%)', fontsize=12)  
    ax1.set_ylabel('Membership Degree', fontsize=12)  
    ax1.set_title('Battery Level Membership Functions', fontsize=14, pad=20)  
    ax1.set_ylim(-0.05, 1.05)  
    ax1.set_xlim(0, 100)  
    
    # 设置电池电量图的刻度  
    ax1.set_xticks([0, 30, 50, 70, 100])  
    ax1.grid(True, linestyle='--', alpha=0.7)  
    ax1.legend(fontsize=10, loc='center right')  
    
    # 绘制计算能力隶属函数  
    ax2.plot(power_x, power_s, 'r-', label='Small (S)', linewidth=2)  
    ax2.plot(power_x, power_m, 'b-', label='Medium (M)', linewidth=2)  
    ax2.plot(power_x, power_l, 'g-', label='Large (L)', linewidth=2)  
    
    ax2.set_xlabel('Computing Power (Geekbench Score)', fontsize=12)  
    ax2.set_ylabel('Membership Degree', fontsize=12)  
    ax2.set_title('Computing Power Membership Functions', fontsize=14, pad=20)  
    ax2.set_ylim(-0.05, 1.05)  
    ax2.set_xlim(400, 2000)  
    
    # 设置计算能力图的刻度  
    ax2.set_xticks([400, 800, 1200, 1600, 2000])  
    ax2.grid(True, linestyle='--', alpha=0.5)  
    ax2.legend(fontsize=10, loc='center right')  
    
    # 调整子图之间的间距  
    plt.tight_layout(pad=2.0)  
    
    # 保存图像，确保高质量输出  
    plt.savefig('membership_functions.png', dpi=300, bbox_inches='tight', facecolor='white')  
    plt.close()

def create_rule_matrix():  
    # 创建图形和轴对象  
    fig, ax = plt.subplots(figsize=(10, 8))  
    
    # 创建规则矩阵数据  
    data = np.array([  
        ['S', 'S', 'M'],  
        ['S', 'M', 'L'],  
        ['M', 'L', 'L']  
    ])  
    
    # 设置背景为白色  
    ax.set_facecolor('white')  
    
    # 设置表格  
    table = ax.table(cellText=data,  
                    cellLoc='center',  
                    loc='center',  
                    cellColours=[['white']*3]*3,  
                    bbox=[0.2, 0.2, 0.6, 0.6])  # [left, bottom, width, height]  
    
    # 设置表格样式  
    table.auto_set_font_size(False)  
    table.set_fontsize(14)  
    for cell in table._cells.values():  
        cell.set_text_props(weight='bold')  
        cell.set_edgecolor('black')  
        cell.set_linewidth(1.5)  
    
    # 设置坐标轴标签  
    ax.set_xticks([])  
    ax.set_yticks([])  
    
    # 添加列标签  
    column_labels = ['S', 'M', 'L']  
    for idx, label in enumerate(column_labels):  
        ax.text(0.3 + idx * 0.2, 0.85, label,   
                ha='center', va='center', fontsize=12)  
    
    # 添加行标签  
    row_labels = ['S', 'M', 'L']  
    for idx, label in enumerate(row_labels):  
        ax.text(0.15, 0.65 - idx * 0.2, label,   
                ha='center', va='center', fontsize=12)  
    
    # 添加标题和轴标签  
    plt.title('Fuzzy Rule Matrix', fontsize=16, pad=20)  
    ax.text(0.5, 0.95, 'Computing Power', ha='center', fontsize=14)  
    ax.text(0.05, 0.5, 'Battery Level', va='center', rotation=90, fontsize=14)  
    
    # 添加图例说明  
    plt.figtext(0.02, 0.02,   
                'S: Small/Low\nM: Medium\nL: Large/High',   
                fontsize=10, ha='left')  
    
    plt.savefig('rule_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.5)  
    plt.close()  

def main():  
    # 生成隶属度函数图  
    plot_membership_functions()  
    
    # 生成规则矩阵图  
    create_rule_matrix()  
    
    print("图像已生成：")  
    print("1. membership_functions.png - 隶属度函数图")  
    print("2. rule_matrix.png - 规则矩阵图")  

if __name__ == "__main__":  
    main()