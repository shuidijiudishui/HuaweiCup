import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import joblib
from pyswarm import pso
from scipy.optimize import minimize

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

# 将磁芯损耗作为目标变量
y = core_loss

# Step 5: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: 构建并训练 LightGBM 模型
lgb_model = LGBMRegressor(n_estimators=1000, random_state=42)
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

# Step 8: 定义目标函数
max_core_loss = max(core_loss)
max_transmission_energy = max(frequency * B_max)

def objective_function(alpha):
    """
    目标函数：最小化磁芯损耗，同时最大化传输磁能（频率 * B_max）。
    我们采用一个加权的方式来结合两个目标。
    """
    # 这里我们采用固定的优化值进行测试
    temp = 70  # 可以根据需求更改
    freq = 300000  # 可以根据需求更改
    B_max_val = B_max.max()  # 可以根据需求更改
    material_code = 0  # 根据需要设置
    waveform_code = 0  # 根据需要设置

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

    # 计算传输磁能
    magnetic_energy = freq * B_max_val
    normalized_energy = magnetic_energy / max_transmission_energy

    # 返回优化目标
    return (1 - alpha) * (core_loss_pred / max_core_loss) - alpha * normalized_energy

# Step 9: 优化 alpha
result = minimize(objective_function, x0=0.5, bounds=[(0, 1)])

# 输出最优的 alpha
best_alpha = result.x[0]
print(f"最优权重 alpha: {best_alpha}")
