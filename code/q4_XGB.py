import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
# 提取温度、频率、B_max 和 材料类型作为输入特征，磁芯损耗为输出

# 提取特征
temperature = all_materials_data.iloc[:, 0].values  # 温度列
frequency = all_materials_data.iloc[:, 1].values    # 频率列
core_loss = all_materials_data.iloc[:, 2].values    # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values     # 励磁波形类型
material = all_materials_data['材料'].values        # 提取材料列

# Step 3: 计算 B_max（忽略非数值列）
# 将磁通密度列（第5列到第1029列）转换为数值，无法转换的值设置为 NaN
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算每行的 B_max（最大值），忽略 NaN
B_max = B_columns.max(axis=1).values

# 对材料类型和励磁波形进行 one-hot 编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')

# 构建特征矩阵
X = np.column_stack((temperature, frequency, B_max))
X = np.hstack((X, material_encoded.values, waveform_encoded.values))

y = core_loss  # 磁芯损耗为目标变量

# Step 4: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: 构建并训练 XGBoost 模型
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
# 打印初始化的参数
print("XGBoost 初始化参数:")
print(xgb_model.get_params())


# Step 6: 模型评估
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

print("训练集误差:")
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("R²:", r2_score(y_train, y_pred_train))

print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='真实值')
plt.scatter(range(len(y_pred_test)), y_pred_test, color='red', label='预测值')
plt.title('XGBoost: 真实值 vs 预测值')
plt.xlabel('样本')
plt.ylabel('磁芯损耗')
plt.legend()
plt.show()

# Step 7: 使用模型对附件三的样本进行预测，并填入附件四
file_test_path = '附件三（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 提取测试集特征
test_temperature = test_data.iloc[:, 0].values
test_frequency = test_data.iloc[:, 1].values
test_waveform = test_data.iloc[:, 3].values   # 励磁波形类型
test_B_max = test_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce').max(axis=1).values  # 计算B_max
test_material = test_data['磁芯材料'].values      # 如果附件三中没有“材料”列，需要手动添加或根据情况处理

# 对材料类型和励磁波形进行 one-hot 编码
test_material_encoded = pd.get_dummies(test_material, prefix='材料')
test_waveform_encoded = pd.get_dummies(test_waveform, prefix='波形')

# 确保测试集的编码列与训练集保持一致
material_columns = material_encoded.columns
waveform_columns = waveform_encoded.columns

for col in material_columns:
    if col not in test_material_encoded:
        test_material_encoded[col] = 0
test_material_encoded = test_material_encoded[material_columns]

for col in waveform_columns:
    if col not in test_waveform_encoded:
        test_waveform_encoded[col] = 0
test_waveform_encoded = test_waveform_encoded[waveform_columns]

# 构建测试集特征矩阵
X_test_predict = np.column_stack((test_temperature, test_frequency, test_B_max))
X_test_predict = np.hstack((X_test_predict, test_material_encoded.values, test_waveform_encoded.values))

# 进行预测
test_predictions = xgb_model.predict(X_test_predict)

# 保留一位小数
test_predictions_rounded = np.round(test_predictions, 1)



# 将结果写入附件四
file_output_path = '附件四.xlsx'
output_data = test_data.copy()  # 假设附件四的结构与附件三相同
output_data['磁芯损耗预测'] = test_predictions_rounded  # 添加预测结果列
output_data.to_excel(file_output_path, index=False)

# 特别输出指定样本序号的预测结果
special_indices = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
special_samples = output_data.iloc[special_indices]
print("\n指定样本序号的磁芯损耗预测结果：")
print(special_samples[['序号', '磁芯损耗预测']])
