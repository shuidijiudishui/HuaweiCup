import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Step 6: 构建自定义损失函数
def custom_loss(y_true, y_pred):
    """自定义损失函数，惩罚负的预测值"""
    residual = y_pred - y_true
    grad = np.where(y_pred < 0, residual + 1000, residual)  # 对负预测值施加惩罚
    hess = np.ones_like(grad)  # Hessian值设为1
    return grad, hess

# Step 7: 构建并训练 LightGBM 模型
lgb_model = LGBMRegressor(n_estimators=1000, random_state=42, objective='regression')

# 训练模型
lgb_model.fit(X_train, y_train)

# 保存 LightGBM 模型
model_filename = 'lgb_model_with_custom_loss.pkl'
joblib.dump(lgb_model, model_filename)

# Step 8: 模型评估
y_pred_train = lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)

# 将负值强制为0
y_pred_train = np.clip(y_pred_train, 0, None)
y_pred_test = np.clip(y_pred_test, 0, None)

# 创建训练集和测试集的对比表格
train_results = pd.DataFrame({
    '实际值': y_train,
    '预测值': y_pred_train
})

test_results = pd.DataFrame({
    '实际值': y_test,
    '预测值': y_pred_test
})

# 输出误差评估
print("训练集误差:")
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("R²:", r2_score(y_train, y_pred_train))

print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

# 输出训练集和测试集的预测结果和实际值
print("\n训练集预测结果与实际值对比:")
print(train_results.head(10))  # 显示前10条

print("\n测试集预测结果与实际值对比:")
print(test_results.head(10))  # 显示前10条

# 将训练集和测试集的预测结果保存到 Excel 中，便于后续查看
train_results.to_excel('训练集预测_vs_实际值.xlsx', index=False)
test_results.to_excel('测试集预测_vs_实际值.xlsx', index=False)

# Step 9: 使用模型对附件三的样本进行预测，并填入附件四
file_test_path = '附件三（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 提取测试集特征
test_temperature = test_data.iloc[:, 1].values
test_frequency = test_data.iloc[:, 2].values
test_waveform = test_data.iloc[:, 4].values
test_B_columns = test_data.iloc[:, 5:1029].apply(pd.to_numeric, errors='coerce')

# 计算测试集的磁通密度特征
test_B_max = test_B_columns.max(axis=1).values
test_B_mean = test_B_columns.mean(axis=1).values
test_B_std = test_B_columns.std(axis=1).values
test_B_peak_to_peak = (test_B_columns.max(axis=1) - test_B_columns.min(axis=1)).values
test_B_skew = test_B_columns.skew(axis=1).values
test_B_kurtosis = test_B_columns.kurtosis(axis=1).values

test_material = test_data['磁芯材料'].values  # 如果附件三中没有“材料”列，需要手动添加或根据情况处理

# 对材料类型和励磁波形进行 one-hot 编码
test_material_encoded = pd.get_dummies(test_material, prefix='磁芯材料')
test_waveform_encoded = pd.get_dummies(test_waveform, prefix='励磁波形')

# 确保测试集的编码列与训练集保持一致
for col in material_encoded.columns:
    if col not in test_material_encoded:
        test_material_encoded[col] = 0
test_material_encoded = test_material_encoded[material_encoded.columns]

for col in waveform_encoded.columns:
    if col not in test_waveform_encoded:
        test_waveform_encoded[col] = 0
test_waveform_encoded = test_waveform_encoded[waveform_encoded.columns]

# 构建测试集特征矩阵
X_test_predict = np.column_stack((test_temperature, test_frequency, test_B_max, test_B_mean, test_B_std, test_B_peak_to_peak, test_B_skew, test_B_kurtosis))
X_test_predict = np.hstack((X_test_predict, test_material_encoded.values, test_waveform_encoded.values))

# 检查训练集和测试集特征矩阵的维度是否一致
print("训练集特征矩阵维度:", X_train.shape)
print("测试集特征矩阵维度:", X_test_predict.shape)

# 进行预测
test_predictions = lgb_model.predict(X_test_predict)

# 将负值强制为0
test_predictions = np.clip(test_predictions, 0, None)

# 保留一位小数
test_predictions_rounded = np.round(test_predictions, 1)

# 将结果写入附件四
file_output_path = '附件四.xlsx'
output_data = test_data.copy()  # 假设附件四的结构与附件三相同
output_data['磁芯损耗预测'] = test_predictions_rounded  # 添加预测结果列
output_data.to_excel(file_output_path, index=False)
