import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: 读取数据
file_path = '附件一（训练集）.xlsx'
materials = ['材料1', '材料2', '材料3', '材料4']
all_materials_data = pd.DataFrame()

# 将四个材料的数据合并，并添加“材料”列
for material in materials:
    material_data = pd.read_excel(file_path, sheet_name=material)
    material_data['材料'] = material
    all_materials_data = pd.concat([all_materials_data, material_data], axis=0)

# Step 2: 数据预处理
# 提取温度、频率、磁芯损耗、励磁波形和材料列
temperature = all_materials_data.iloc[:, 0].values  # 温度列
frequency = all_materials_data.iloc[:, 1].values    # 频率列
core_loss = all_materials_data.iloc[:, 2].values    # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values     # 励磁波形类型列
material = all_materials_data['材料'].values        # 材料列

# 将所有的磁通密度列转换为数值类型，并忽略非数值数据
B_max_data = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算每一行的最大值
B_max = B_max_data.max(axis=1).fillna(0).values

# 将材料类型和励磁波形类型进行one-hot编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')

# 构建特征矩阵
X = np.vstack([temperature, frequency, B_max]).T
X = np.concatenate([X, material_encoded, waveform_encoded], axis=1)
y = core_loss

# Step 3: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: 构建并训练支持向量回归模型
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

# Step 6: 进行预测
y_pred_train = svr_model.predict(X_train_scaled)
y_pred_test = svr_model.predict(X_test_scaled)

# Step 7: 模型评估
print("训练集误差:")
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("R²:", r2_score(y_train, y_pred_train))

print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

# Step 8: 使用模型对附件三的样本进行预测，并填入附件四
file_test_path = '附件三（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 读取样本序号、温度、频率、B_max等特征
test_temperature = test_data.iloc[:, 0].values
test_frequency = test_data.iloc[:, 1].values
test_material = test_data.iloc[:, 3].values
test_waveform = test_data.iloc[:, 2].values

# 计算B_max时同样确保处理数值类型的数据
test_B_max = pd.to_numeric(test_data.iloc[:, 4:1029].max(axis=1), errors='coerce').fillna(0).values

# 将材料类型和励磁波形类型进行one-hot编码
test_material_encoded = pd.get_dummies(test_material, prefix='材料')
test_waveform_encoded = pd.get_dummies(test_waveform, prefix='波形')

# 构建测试集特征矩阵
X_test_predict = np.vstack([test_temperature, test_frequency, test_B_max]).T
X_test_predict = np.concatenate([X_test_predict, test_material_encoded, test_waveform_encoded], axis=1)

# 进行标准化
X_test_predict_scaled = scaler.transform(X_test_predict)

# 进行预测
test_predictions = svr_model.predict(X_test_predict_scaled)

# 保留一位小数
test_predictions_rounded = np.round(test_predictions, 1)

# 将结果写入附件四
file_output_path = '附件四.xlsx'
output_data = pd.read_excel(file_output_path)
output_data['磁芯损耗预测'] = test_predictions_rounded  # 假设第3列为磁芯损耗预测列
output_data.to_excel(file_output_path, index=False)

# 特别输出特定样本序号的预测结果
special_indices = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
special_results = output_data.iloc[special_indices, :]
print(special_results[['样本序号', '磁芯损耗预测']])
