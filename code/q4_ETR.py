import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Step 3: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 构建并训练极端随机树模型
etr_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr_model.fit(X_train, y_train)

# Step 5: 模型评估
y_pred_train = etr_model.predict(X_train)
y_pred_test = etr_model.predict(X_test)

# 评估模型在训练集和测试集上的性能
print("训练集误差:")
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("R²:", r2_score(y_train, y_pred_train))

print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

# Step 6: 绘制真实值和预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='真实值')
plt.scatter(range(len(y_pred_test)), y_pred_test, color='red', label='预测值')
plt.title('极端随机树: 真实值 vs 预测值')
plt.xlabel('样本')
plt.ylabel('磁芯损耗')
plt.legend()
plt.show()

# Step 7: 使用模型对附件三的样本进行预测，并填入附件四
file_test_path = '附件三（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 读取样本序号、温度、频率、B_max等特征
test_temperature = test_data.iloc[:, 0].values
test_frequency = test_data.iloc[:, 1].values
test_waveform = test_data.iloc[:, 3].values
test_B_max = test_data.iloc[:, 4:1029].max(axis=1).values

# 对励磁波形进行 one-hot 编码
test_waveform_encoded = pd.get_dummies(test_waveform)

# 构建测试集特征矩阵
X_test_predict = np.vstack([test_temperature, test_frequency, test_B_max]).T
X_test_predict = np.concatenate([X_test_predict, test_waveform_encoded], axis=1)

# 进行预测
test_predictions = etr_model.predict(X_test_predict)

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
