import pandas as pd
import numpy as np

# 读取数据
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

# 提取励磁波形和磁通密度数据
waveform = all_materials_data.iloc[:, 3].values  # 励磁波形类型
flux_density_data = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce').values  # 磁通密度数据（1024采样点）


# 定义一个函数来计算特征：最大值、最小值、斜率变化数量
def calculate_waveform_features(waveform_data):
    # 找到最大值数量
    num_max_values = np.sum(waveform_data == np.max(waveform_data))

    # 找到最小值数量
    num_min_values = np.sum(waveform_data == np.min(waveform_data))

    # 计算斜率变化数量（用相邻数据点的差值来判断）
    diff = np.diff(waveform_data)
    slope_changes = np.sum(np.abs(np.diff(np.sign(diff))))
    num_slope_changes = slope_changes

    return num_max_values, num_min_values, num_slope_changes


# 初始化字典来存储不同波形的特征
waveform_features = {
    '正弦波': {'最大值数量': [], '最小值数量': [], '斜率变化数量': []},
    '三角波': {'最大值数量': [], '最小值数量': [], '斜率变化数量': []},
    '梯形波': {'最大值数量': [], '最小值数量': [], '斜率变化数量': []}
}

# 统计不同波形对应的特征数量
for i in range(len(flux_density_data)):
    waveform_type = waveform[i]  # 当前样本的波形类型
    flux_density = flux_density_data[i, :]

    # 跳过包含 NaN 的样本
    if np.isnan(flux_density).any():
        continue

    # 计算该样本的特征
    num_max_values, num_min_values, num_slope_changes = calculate_waveform_features(flux_density)

    # 将结果存储到对应波形的列表中
    if waveform_type in waveform_features:
        waveform_features[waveform_type]['最大值数量'].append(num_max_values)
        waveform_features[waveform_type]['最小值数量'].append(num_min_values)
        waveform_features[waveform_type]['斜率变化数量'].append(num_slope_changes)

# 输出每种波形的特征统计结果
for wave_type, features in waveform_features.items():
    print(f"\n波形类型: {wave_type}")
    print(f"最大值数量平均值: {np.mean(features['最大值数量'])}")
    print(f"最小值数量平均值: {np.mean(features['最小值数量'])}")
    print(f"斜率变化数量平均值: {np.mean(features['斜率变化数量'])}")

# 将结果保存到 Excel 文件中
result_df = pd.DataFrame({
    '波形类型': list(waveform_features.keys()),
    '最大值数量平均值': [np.mean(waveform_features[w]['最大值数量']) for w in waveform_features],
    '最小值数量平均值': [np.mean(waveform_features[w]['最小值数量']) for w in waveform_features],
    '斜率变化数量平均值': [np.mean(waveform_features[w]['斜率变化数量']) for w in waveform_features],
})

# 保存到 Excel 文件中
result_df.to_excel('简化波形特征统计结果.xlsx', index=False)
