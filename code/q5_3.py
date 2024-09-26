import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from statsmodels.sandbox.distributions.examples.ex_gof import freq

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

B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算 B_max, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis
B_max = B_columns.max(axis=1)
B_mean = B_columns.mean(axis=1)
B_std = B_columns.std(axis=1)
B_peak_to_peak = B_columns.max(axis=1) - B_columns.min(axis=1)
B_skew = pd.DataFrame(B_columns).skew(axis=1).values  # 使用 pandas 计算偏度
B_kurtosis = pd.DataFrame(B_columns).kurtosis(axis=1).values  # 使用 pandas 计算峰度

# 获取每个特征的最小值和最大值
min_B_max, max_B_max = B_max.min(), B_max.max()
min_B_mean, max_B_mean = B_mean.min(), B_mean.max()
min_B_std, max_B_std = B_std.min(), B_std.max()
min_B_peak_to_peak, max_B_peak_to_peak = B_peak_to_peak.min(), B_peak_to_peak.max()
min_B_skew, max_B_skew = B_skew.min(), B_skew.max()
min_B_kurtosis, max_B_kurtosis = B_kurtosis.min(), B_kurtosis.max()

# 添加计算出来的特征到 DataFrame
all_materials_data['B_max'] = B_max
all_materials_data['B_mean'] = B_mean
all_materials_data['B_std'] = B_std
all_materials_data['B_peak_to_peak'] = B_peak_to_peak
all_materials_data['B_skew'] = B_skew
all_materials_data['B_kurtosis'] = B_kurtosis

# 编码材料类型和波形类型
all_materials_data['材料编码'] = all_materials_data['材料'].astype('category').cat.codes
all_materials_data['波形编码'] = all_materials_data['励磁波形'].astype('category').cat.codes

# 定义模型的输入和输出
X = all_materials_data[['温度，oC', '频率，Hz', 'B_max', '波形编码', '材料编码', 'B_mean', 'B_std', 'B_peak_to_peak', 'B_skew', 'B_kurtosis']]
y = all_materials_data['磁芯损耗，w/m3']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 LightGBM 模型
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'seed': 42
}

# 训练模型
gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=100)

# Step 2: 定义差分进化算法的目标函数
def objective_function(individual):
    # 提取个体的特征：温度、频率、磁通密度峰值、波形类型、材料，以及磁通密度统计特征
    temp = individual[0]  # 温度
    freq = individual[1]  # 频率
    B_max = individual[2]  # 磁通密度峰值
    waveform_code = int(individual[3])  # 励磁波形类型编码 (正弦波、三角波、梯形波)
    material_code = int(individual[4])  # 材料编码

    # 额外的磁通密度分布特征
    B_mean = individual[5]  # B_mean
    B_std = individual[6]  # B_std
    B_peak_to_peak = individual[7]  # B_peak_to_peak
    B_skew = individual[8]  # B_skew
    B_kurtosis = individual[9]  # B_kurtosis

    # 将个体的特征拼接为模型输入
    feature_input = np.hstack([temp, freq, B_max, waveform_code, material_code, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis])

    # 使用 LightGBM 模型预测磁芯损耗
    core_loss = gbm.predict([feature_input])[0]

    # # 保证磁芯损耗为非负
    # core_loss = max(core_loss, 0)

    # 计算传输磁能
    transmitted_energy = freq * B_max

    # 计算传输磁能
    magnetic_energy = freq * B_max
    normalized_energy = magnetic_energy / max_transmission_energy
    alpha = 0.9

    # 差分进化是最小化算法：我们希望最小化磁芯损耗，最大化传输磁能（因此取负传输磁能）
    return (1 - alpha) * (core_loss / max_core_loss) - alpha * normalized_energy

max_core_loss = max(gbm.predict())  # 或根据模型预测的最大值
max_transmission_energy = max(freq * B_max)

# Step 3: 设置变量的边界 [温度, 频率, 磁通密度峰值, 波形类型, 材料, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis]
bounds = [
    (25, 90),  # 温度范围 (25 °C 到 90 °C)
    (50000, 500000),  # 频率范围 (50 kHz 到 500 kHz)
    (min_B_max, max_B_max),  # B_max 范围 (磁通密度峰值)
    (0, 2),  # 波形类型 (正弦波、三角波、梯形波) 编码为 0, 1, 2
    (0, 3),  # 材料编码 (材料1, 材料2, 材料3, 材料4) 编码为 0, 1, 2, 3
    (min_B_mean, max_B_mean),  # B_mean 边界
    (min_B_std, max_B_std),  # B_std 边界
    (min_B_peak_to_peak, max_B_peak_to_peak),  # B_peak_to_peak 边界
    (min_B_skew, max_B_skew),  # B_skew 边界
    (min_B_kurtosis, max_B_kurtosis)  # B_kurtosis 边界
]

# Step 4: 使用差分进化算法进行优化
result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=1000, popsize=15,
                                mutation=(0.5, 1), recombination=0.7)

# Step 5: 输出优化结果
optimal_solution = result.x

# 预测最小磁芯损耗
optimal_temp = optimal_solution[0]
optimal_freq = optimal_solution[1]
optimal_B_max = optimal_solution[2]
optimal_waveform = optimal_solution[3]
optimal_material = int(optimal_solution[4])
optimal_B_mean = optimal_solution[5]
optimal_B_std = optimal_solution[6]
optimal_B_peak_to_peak = optimal_solution[7]
optimal_B_skew = optimal_solution[8]
optimal_B_kurtosis = optimal_solution[9]

# 构造用于 LightGBM 模型的输入特征
optimal_features = np.hstack([optimal_temp, optimal_freq, optimal_B_max, optimal_waveform, optimal_material,
                              optimal_B_mean, optimal_B_std, optimal_B_peak_to_peak, optimal_B_skew, optimal_B_kurtosis])

# 预测最小磁芯损耗
min_core_loss = gbm.predict([optimal_features])[0]
min_core_loss = max(min_core_loss, 0)  # 确保损耗为非负

# 计算对应的最大传输磁能
max_transmitted_energy = optimal_freq * optimal_B_max

# 输出最优解、最小磁芯损耗、最大传输磁能
print(f"最优解: 温度 = {optimal_temp:.2f} °C, 频率 = {optimal_freq:.2f} Hz, B_max = {optimal_B_max:.4f} T, "
      f"波形编码 = {optimal_waveform:.0f}, 材料编码 = {optimal_material}")
print(f"最小磁芯损耗: {min_core_loss:.4f} W/m³")
print(f"最大传输磁能: {max_transmitted_energy:.4f}")
