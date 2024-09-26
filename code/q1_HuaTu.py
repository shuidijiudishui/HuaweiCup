import pandas as pd
import matplotlib.pyplot as plt

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

# Step 2: 提取励磁波形列
waveform = all_materials_data.iloc[:, 3].values  # 励磁波形类型列

# Step 3: 统计每种波形类型的数量
waveform_counts = pd.Series(waveform).value_counts()

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑

# Step 4: 绘制条形图
plt.figure(figsize=(8, 6))
waveform_counts.plot(kind='bar', color=['blue', 'green', 'orange'])

# 添加标题和标签
plt.title('不同励磁波形对应数据数量', fontsize=16)
plt.xlabel('励磁波形', fontsize=14)
plt.ylabel('数据数量', fontsize=14)

# 显示图表
plt.xticks(rotation=0)
plt.show()
