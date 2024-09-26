import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
import joblib
from pyswarm import pso
from scipy import stats


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
temperature = all_materials_data.iloc[:, 0].values  # 温度列
frequency = all_materials_data.iloc[:, 1].values  # 频率列
core_loss = all_materials_data.iloc[:, 2].values  # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values  # 励磁波形类型
material = all_materials_data['材料'].values  # 提取材料列

# Step 3: 计算磁通密度分布的统计特征
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')
B_max = B_columns.max(axis=1).values

# Step 4: 对材料类型和励磁波形进行 one-hot 编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')

# 构建特征矩阵
X = np.column_stack((temperature, frequency, B_max))
X = np.hstack((X, material_encoded.values, waveform_encoded.values))

y = core_loss

# 确保特征矩阵没有 NaN
X = np.nan_to_num(X)  # 用0替代NaN，如果需要用其他值请自行调整

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# 其余代码保持不变


# Step 6: 构建并训练 LightGBM 模型
lgb_model = XGBRegressor(n_estimators=200, random_state=42)
lgb_model.fit(X_train, y_train)

# 保存 LightGBM 模型
model_filename = 'lgb_model_filtered.pkl'
joblib.dump(lgb_model, model_filename)

# Step 7: 模型评估
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)
print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

# Step 8: 定义优化目标函数
def pso_objective_function(x):
    """
    目标函数：最小化磁芯损耗，同时最大化传输磁能（频率 * B_max）。
    我们采用一个加权的方式来结合两个目标。
    """
    # 解压优化变量
    temp, freq, B_max_val, material_code, waveform_code = x

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
        [temp, freq, B_max_val],
        material_one_hot,
        waveform_one_hot
    )).reshape(1, -1)

    # 预测磁芯损耗
    core_loss_pred = lgb_model.predict(feature_input)[0]
    print('core_loss_pred', core_loss_pred)

    # 计算传输磁能
    magnetic_energy = freq * B_max_val
    normalized_energy = magnetic_energy / max_transmission_energy

    # 定义目标函数权重
    alpha = 0.9  # 权重可根据需要调整

    # 最小化磁芯损耗并最大化传输磁能
    return (1 - alpha) * (core_loss_pred / max_core_loss) - alpha * normalized_energy

max_core_loss = max(core_loss)  # 或根据模型预测的最大值
max_transmission_energy = max(frequency * B_max)

# Step 9: 定义变量边界
lower_bounds = [
    25,  # 温度下限
    50000,  # 频率下限
    B_max.min(),  # B_max下限
    0,  # 材料编码下限
    0  # 波形编码下限
]

upper_bounds = [
    90,  # 温度上限
    500000,  # 频率上限
    B_max.max(),  # B_max上限
    material_encoded.shape[1] - 1,  # 材料编码上限
    waveform_encoded.shape[1] - 1  # 波形编码上限
]

# 执行粒子群优化
optimal_values, optimal_loss = pso(pso_objective_function, lower_bounds, upper_bounds, swarmsize=1000, maxiter=1000)

# 输出优化结果
optimal_temp, optimal_freq, optimal_B_max, optimal_material, optimal_waveform = optimal_values

print("\n粒子群优化后的参数：")
print(f"温度: {optimal_temp:.2f} °C")
print(f"频率: {optimal_freq:.2f} Hz")
print(f"B_max: {optimal_B_max:.4f} T")
print(f"材料编码: {int(round(optimal_material))}")
print(f"波形编码: {int(round(optimal_waveform))}")

# 解释材料和波形编码
material_mapping = {i: col for i, col in enumerate(material_encoded.columns)}
waveform_mapping = {i: col for i, col in enumerate(waveform_encoded.columns)}

print(f"材料类型: {material_mapping[int(round(optimal_material))]}")
print(f"励磁波形类型: {waveform_mapping[int(round(optimal_waveform))]}")

# Step 10: 基于最优解计算最小磁芯损耗和最大传输磁能
material_one_hot = np.zeros(material_encoded.shape[1])
material_one_hot[int(round(optimal_material))] = 1  # 将对应材料位置置1

waveform_one_hot = np.zeros(waveform_encoded.shape[1])
waveform_one_hot[int(round(optimal_waveform))] = 1  # 将对应波形位置置1

optimal_feature_input = np.concatenate((
    [optimal_temp, optimal_freq, optimal_B_max],
    material_one_hot,
    waveform_one_hot
)).reshape(1, -1)

# 使用 LightGBM 模型预测磁芯损耗
min_core_loss = lgb_model.predict(optimal_feature_input)[0]

# 计算最大传输磁能（频率 * B_max）
max_magnetic_energy = optimal_freq * optimal_B_max

print("\n粒子群优化后的结果：")
print(f"最小磁芯损耗: {min_core_loss:.4f} W/m^3")
print(f"最大传输磁能: {max_magnetic_energy:.4f} Hz*T")
