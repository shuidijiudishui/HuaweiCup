import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Excel 文件中的所有表
file_path = '附件一（训练集）.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# 创建一个字典来存储每个表的数据
data_dict = {}

# 统计每种材料的行数
material_counts = {}

for sheet in sheet_names:
    data_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)
    material_counts[sheet] = data_dict[sheet].shape[0]  # 统计行数

# 设置可视化风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 创建 2x2 的图形布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# 1. 磁芯材料的数量分布
sns.barplot(x=list(material_counts.keys()), y=list(material_counts.values()), palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('不同磁芯材料的数量分布')
axes[0, 0].set_xlabel('磁芯材料')
axes[0, 0].set_ylabel('数量')
axes[0, 0].grid(axis='y')

# 2. 温度的数量分布
temperature_data = pd.concat([df['温度，oC'] for df in data_dict.values()], axis=0)
sns.countplot(x=temperature_data, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('不同温度的数量分布')
axes[0, 1].set_xlabel('温度 (°C)')
axes[0, 1].set_ylabel('数量')
axes[0, 1].grid(axis='y')

# 3. 频率的数量分布
frequency_data = pd.concat([df['频率，Hz'] for df in data_dict.values()], axis=0)
sns.histplot(frequency_data, bins=20, kde=False, color='skyblue', ax=axes[1, 0])
axes[1, 0].set_title('不同频率的数量分布')
axes[1, 0].set_xlabel('频率 (Hz)')
axes[1, 0].set_ylabel('数量')
axes[1, 0].grid(axis='y')

# 4. 励磁波形的数量分布
waveform_data = pd.concat([df['励磁波形'] for df in data_dict.values()], axis=0)
sns.countplot(x=waveform_data, palette='viridis', ax=axes[1, 1])
axes[1, 1].set_title('不同励磁波形的数量分布')
axes[1, 1].set_xlabel('励磁波形')
axes[1, 1].set_ylabel('数量')
axes[1, 1].grid(axis='y')

plt.show()
