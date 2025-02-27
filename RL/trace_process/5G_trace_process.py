import pandas as pd
import os

# 源目录和目标目录
source_dir = '/home/zlp/lfbm-ppo/data/bandwidth/5G-production-dataset/5G-production-dataset/Netflix/Static/Season3-StrangerThings'
target_dir = '/home/zlp/lfbm-ppo/data/bandwidth/5G-production-dataset/5G-production-dataset/Netflix/Static/Accessed-Season3-StrangerThings'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 处理每个CSV文件
for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        # 读取CSV文件
        df = pd.read_csv(os.path.join(source_dir, filename))
        
        # 筛选DL_bitrate大于10000的行
        filtered_df = df[df['DL_bitrate'] > 10000].copy()
        
        if not filtered_df.empty:
            # 创建新的Timestamp列，从0开始每次加5
            filtered_df['Timestamp'] = range(0, len(filtered_df) * 5, 5)
            
            # 将DL_bitrate乘以1000
            filtered_df['DL_bitrate'] = filtered_df['DL_bitrate'] * 1000
            
            # 只保留Timestamp和DL_bitrate列
            filtered_df = filtered_df[['Timestamp', 'DL_bitrate']]
            
            # 将文件名改为.log后缀
            new_filename = os.path.splitext(filename)[0] + '.log'
            
            # 保存为空格分隔的文本文件，不带标题行
            filtered_df.to_csv(
                os.path.join(target_dir, new_filename), 
                sep=' ', 
                header=False, 
                index=False
            )
            
            # 打印处理信息
            print(f"处理文件 {filename}: 保留了 {len(filtered_df)} 行数据")