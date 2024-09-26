import pandas as pd
import numpy as np

# Step 1: 读取数据，并根据 sheet 名添加“材料”列
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']

# 初始化列表以存储所有数据
all_data = []

# 遍历每个 sheet（代表不同的材料）
for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['材料'] = sheet  # 添加“材料”列，值为 sheet 名
    all_data.append(data)

# 将所有材料的数据合并为一个 DataFrame
all_materials_data = pd.concat(all_data, ignore_index=True)

# Step 2: 数据预处理
# 提取频率列
frequency = all_materials_data.iloc[:, 1].values    # 频率列

# 提取磁通密度数据 (第5列到第1029列)
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算峰值磁通密度 (B_max)
B_max = B_columns.max(axis=1).values

# Step 3: 计算传输磁能
# 传输磁能等于 频率 * 峰值磁通密度
transmission_energy = frequency * B_max

# 将传输磁能加入到 DataFrame 中
all_materials_data['传输磁能'] = transmission_energy

# Step 4: 保存结果到新的 Excel 文件
output_file_path = '传输磁能计算结果.xlsx'
all_materials_data.to_excel(output_file_path, index=False)

print("传输磁能计算已完成，结果已保存到:", output_file_path)
