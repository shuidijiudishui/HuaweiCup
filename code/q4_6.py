import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib


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
frequency = all_materials_data.iloc[:, 1].values    # 频率列
core_loss = all_materials_data.iloc[:, 2].values    # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values     # 励磁波形类型
material = all_materials_data['材料'].values        # 提取材料列

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
print("维度", X.shape)

# 将磁芯损耗作为目标变量
y = core_loss

# Step 5: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: 构建并训练 LightGBM 模型
lgb_model = LGBMRegressor(n_estimators=1000, random_state=42)
lgb_model.fit(X_train, y_train)


# Step 6: 定义 LightGBM 超参数优化的目标函数
def lgb_evaluate(params):
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )

    # 使用交叉验证计算模型的负均方误差（MSE）
    mse = -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))

    return mse


# Step 7: 设置差分进化算法的超参数搜索范围
param_bounds = [
    (100, 1000),  # n_estimators 范围
    (0.01, 0.3),  # learning_rate 范围
    (3, 10)  # max_depth 范围
]

# Step 8: 调用差分进化算法进行超参数优化
result = differential_evolution(lgb_evaluate, param_bounds, maxiter=10, seed=42)

# 输出优化结果
best_params = result.x
print("最优超参数: ", best_params)

# Step 9: 使用最优超参数训练 LightGBM 模型
best_n_estimators = int(best_params[0])
best_learning_rate = best_params[1]
best_max_depth = int(best_params[2])

lgb_model_optimized = LGBMRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    random_state=42
)

lgb_model_optimized.fit(X_train, y_train)

# Step 10: 评估模型性能
y_pred = lgb_model_optimized.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("优化后的 LightGBM 模型 MSE: ", mse)