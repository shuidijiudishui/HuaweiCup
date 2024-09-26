import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib
from scipy.optimize import dual_annealing
from xgboost import XGBRegressor


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
# 提取温度、频率、磁芯损耗列和波形列
temperature = all_materials_data.iloc[:, 0].values  # 温度列
frequency = all_materials_data.iloc[:, 1].values  # 频率列
core_loss = all_materials_data.iloc[:, 2].values  # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values  # 励磁波形类型
material = all_materials_data['材料'].values  # 提取材料列

# Step 3: 计算磁通密度分布的统计特征
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算峰值磁通密度 (B_max)
B_max = B_columns.max(axis=1).values

# 计算磁通密度的均值、标准差、峰峰值、偏度、峰度
B_mean = B_columns.mean(axis=1).values
B_std = B_columns.std(axis=1).values
B_peak_to_peak = (B_columns.max(axis=1) - B_columns.min(axis=1)).values
B_skew = B_columns.skew(axis=1).values
B_kurtosis = B_columns.kurtosis(axis=1).values

# Step 4: 对材料类型和励磁波形进行 one-hot 编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')

# 构建特征矩阵，包括原始的温度、频率、B_max，以及提取的磁通密度分布特征
X = np.column_stack((temperature, frequency, B_max, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis))
X = np.hstack((X, material_encoded.values, waveform_encoded.values))
print("特征矩阵维度:", X.shape)

# 将磁芯损耗作为目标变量
y = core_loss

# Step 5: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: 构建并训练 LightGBM 模型
lgb_model = LGBMRegressor(n_estimators=1000, random_state=42)
lgb_model.fit(X_train, y_train)

# 保存 LightGBM 模型
model_filename = 'lgb_model.pkl'
joblib.dump(lgb_model, model_filename)


# Step 7: 定义优化目标函数
def objective_function(x):
    """
    目标函数：最小化磁芯损耗，同时最大化传输磁能（频率 * B_max）。
    我们采用一个加权的方式来结合两个目标。
    """
    # 解压优化变量
    temp, freq, B_max_val, B_mean_val, B_std_val, B_ptp_val, B_skew_val, B_kurt_val, material_code, waveform_code = x

    # 强制将材料和波形编码转换为整数
    material_code = int(round(material_code))
    waveform_code = int(round(waveform_code))

    # 确保编码在有效范围内
    material_code = max(0, min(material_code, material_encoded.shape[1] - 1))
    waveform_code = max(0, min(waveform_code, waveform_encoded.shape[1] - 1))

    # 创建 One-Hot 编码
    material_one_hot = np.zeros(material_encoded.shape[1])
    waveform_one_hot = np.zeros(waveform_encoded.shape[1])
    material_one_hot[material_code] = 1
    waveform_one_hot[waveform_code] = 1

    # 构建特征向量
    feature_input = np.concatenate((
        [temp, freq, B_max_val, B_mean_val, B_std_val, B_ptp_val, B_skew_val, B_kurt_val],
        material_one_hot,
        waveform_one_hot
    )).reshape(1, -1)

    # 预测磁芯损耗
    core_loss_pred = lgb_model.predict(feature_input)[0]

    # 计算传输磁能
    magnetic_energy = freq * B_max_val
    normalized_energy = magnetic_energy / max_transmission_energy
    # 定义目标函数权重
    alpha = 0.95 # 权重可根据需要调整

    # 我们希望最小化磁芯损耗并最大化传输磁能，因此定义目标为：
    # minimize(core_loss_pred - alpha * magnetic_energy)
    # return core_loss_pred - alpha * magnetic_energy
# 最小化磁芯损耗并最大化传输磁能
    return (1 - alpha) * (core_loss_pred / max_core_loss) - alpha * normalized_energy

max_core_loss = max(core_loss)  # 或根据模型预测的最大值
max_transmission_energy = max(frequency * B_max)

# Step 8: 定义变量边界
bounds = [
    (25, 90),  # 温度范围
    (50000, 500000),  # 频率范围
    (B_max.min(), B_max.max()),  # B_max范围
    (B_mean.min(), B_mean.max()),  # B_mean范围
    (B_std.min(), B_std.max()),  # B_std范围
    (B_peak_to_peak.min(), B_peak_to_peak.max()),  # B_peak_to_peak范围
    (B_skew.min(), B_skew.max()),  # B_skew范围
    (B_kurtosis.min(), B_kurtosis.max()),  # B_kurtosis范围
    (0, material_encoded.shape[1] - 1),  # 材料编码范围 (0 to 3)
    (0, waveform_encoded.shape[1] - 1)  # 波形编码范围 (0 to 2)
]

# Step 9: 执行模拟退火优化
# 初始猜测
x0 = [
    50,  # 温度
    200000,  # 频率
    np.mean(B_max),  # B_max
    np.mean(B_mean),  # B_mean
    np.mean(B_std),  # B_std
    np.mean(B_peak_to_peak),  # B_peak_to_peak
    np.mean(B_skew),  # B_skew
    np.mean(B_kurtosis),  # B_kurtosis
    0,  # 材料编码初始值
    0  # 波形编码初始值
]

# 执行模拟退火
result = dual_annealing(objective_function, bounds, x0=x0, maxiter=1000, seed=42)

# 输出优化结果
optimal_values = result.x
optimal_temp, optimal_freq, optimal_B_max, optimal_B_mean, optimal_B_std, optimal_B_ptp, optimal_B_skew, optimal_B_kurt, optimal_material, optimal_waveform = optimal_values

print("\n模拟退火优化后的参数：")
print(f"温度: {optimal_temp:.2f} °C")
print(f"频率: {optimal_freq:.2f} Hz")
print(f"B_max: {optimal_B_max:.4f} T")
print(f"B_mean: {optimal_B_mean:.4f} T")
print(f"B_std: {optimal_B_std:.4f} T")
print(f"B_peak_to_peak: {optimal_B_ptp:.4f} T")
print(f"B_skew: {optimal_B_skew:.4f}")
print(f"B_kurtosis: {optimal_B_kurt:.4f}")
print(f"材料编码: {int(round(optimal_material))}")
print(f"波形编码: {int(round(optimal_waveform))}")

# 解释材料和波形编码
material_mapping = {i: col for i, col in enumerate(material_encoded.columns)}
waveform_mapping = {i: col for i, col in enumerate(waveform_encoded.columns)}

print(f"材料类型: {material_mapping[int(round(optimal_material))]}")
print(f"励磁波形类型: {waveform_mapping[int(round(optimal_waveform))]}")

material_one_hot = np.zeros(4)
material_one_hot[int(round(optimal_material))] = 1  # 将对应材料位置置1

waveform_one_hot = np.zeros(3)
waveform_one_hot[int(round(optimal_waveform))] = 1  # 将对应波形位置置1

# Step 10: 基于最优解计算最小磁芯损耗和最大传输磁能
optimal_feature_input = np.concatenate((
    [optimal_temp, optimal_freq, optimal_B_max, optimal_B_mean, optimal_B_std, optimal_B_ptp, optimal_B_skew, optimal_B_kurt],
    material_one_hot,
    waveform_one_hot
)).reshape(1, -1)

# 使用 LightGBM 模型预测磁芯损耗
min_core_loss = lgb_model.predict(optimal_feature_input)[0]

# 计算最大传输磁能（频率 * B_max）
max_magnetic_energy = optimal_freq * optimal_B_max

print("\n模拟退火优化后的结果：")
print(f"最小磁芯损耗: {min_core_loss:.4f} W/m^3")
print(f"最大传输磁能: {max_magnetic_energy:.4f} Hz*T")