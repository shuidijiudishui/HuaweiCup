import pandas as pd
import numpy as np

# 1. 数据加载
file_path = '附件一（训练集）.xlsx'
xls = pd.ExcelFile(file_path)

# 读取所有表格并合并
data_frames = []
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df['磁芯材料'] = sheet_name  # 添加材料列
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)

# 2. 数据预处理：提取需要的列，并进行标准化
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# 确保列名与数据一致
data['温度'] = normalize(data['温度，oC'])
data['频率'] = normalize(data['频率，Hz'])
data['磁芯损耗'] = normalize(data['磁芯损耗，w/m3'])

# 对于励磁波形进行独热编码
data_encoded = pd.get_dummies(data, columns=['励磁波形'])

# 3. 计算灰色关联度
def grey_relational_coefficient(x, y, rho=0.5):
    diff = np.abs(x - y)
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    coefficient = (min_diff + rho * max_diff) / (diff + rho * max_diff)
    return np.mean(coefficient)

# 计算关联度
associations = {}
# 选择分析的列
for col in ['温度', '频率'] + list(data_encoded.columns[data_encoded.columns.str.contains('励磁波形')]) + ['磁芯材料']:
    associations[col] = grey_relational_coefficient(data_encoded[col], data_encoded['磁芯损耗'])

# 4. 结果分析
associations_df = pd.DataFrame(list(associations.items()), columns=['因素', '关联度'])
print(associations_df)

# 讨论：根据关联度的结果分析各因素对磁芯损耗的影响程度

