import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Step 1: 读取数据，并根据 sheet 名添加“材料”列
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']

all_data = []
for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['材料'] = sheet
    all_data.append(data)

all_materials_data = pd.concat(all_data, ignore_index=True)
# 删除包含 NaN 的行
all_materials_data = all_materials_data.dropna()

# Step 2: 数据预处理
temperature = all_materials_data.iloc[:, 0].values
frequency = all_materials_data.iloc[:, 1].values
core_loss = all_materials_data.iloc[:, 2].values
waveform = all_materials_data.iloc[:, 3].values
material = all_materials_data['材料'].values

# Step 3: 计算 B_max 和其他特征
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')
B_max = B_columns.max(axis=1).values
B_mean = B_columns.mean(axis=1).values
B_std = B_columns.std(axis=1).values
B_peak_to_peak = B_max - B_columns.min(axis=1).values
B_skew = B_columns.apply(stats.skew, axis=1).values
B_kurtosis = B_columns.apply(stats.kurtosis, axis=1).values

# # 傅里叶特征
# B_fft = np.fft.fft(B_columns, axis=1)
# B_fft_magnitude = np.abs(B_fft[:, :B_fft.shape[1] // 2])
# fft_mean = B_fft_magnitude.mean(axis=1)
# fft_std = B_fft_magnitude.std(axis=1)

# 对材料类型和励磁波形进行 one-hot 编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')

# 构建特征矩阵
X = np.column_stack((temperature, frequency, B_max, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis))
X = np.hstack((X, material_encoded.values, waveform_encoded.values))

y = core_loss
# 确保特征矩阵没有 NaN
X = np.nan_to_num(X)  # 用0替代NaN，如果需要用其他值请自行调整

# Step 4: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: 构建并训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: 模型评估
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

print("训练集误差:")
print("MSE:", mean_squared_error(y_train, y_pred_train))
print("MAE:", mean_absolute_error(y_train, y_pred_train))
print("R²:", r2_score(y_train, y_pred_train))

print("\n测试集误差:")
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("R²:", r2_score(y_test, y_pred_test))

# Step 7: 使用模型对附件三的样本进行预测
file_test_path = '附件三（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 提取测试集特征
test_temperature = test_data.iloc[:, 0].values
test_frequency = test_data.iloc[:, 1].values
test_waveform = test_data.iloc[:, 3].values
test_B_columns = test_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算测试集特征
test_B_max = test_B_columns.max(axis=1).values
test_B_mean = test_B_columns.mean(axis=1).values
test_B_std = test_B_columns.std(axis=1).values
test_B_peak_to_peak = test_B_max - test_B_columns.min(axis=1).values
test_B_skew = test_B_columns.apply(stats.skew, axis=1).values
test_B_kurtosis = test_B_columns.apply(stats.kurtosis, axis=1).values
test_B_fft = np.fft.fft(test_B_columns, axis=1)
test_B_fft_magnitude = np.abs(test_B_fft[:, :test_B_fft.shape[1] // 2])
# test_fft_mean = test_B_fft_magnitude.mean(axis=1)
# test_fft_std = test_B_fft_magnitude.std(axis=1)

# 对材料类型和励磁波形进行 one-hot 编码
test_material_encoded = pd.get_dummies(test_data['材料'], prefix='材料')
test_waveform_encoded = pd.get_dummies(test_waveform, prefix='波形')

# 构建测试集特征矩阵
X_test_predict = np.column_stack((test_temperature, test_frequency, test_B_max, test_B_mean, test_B_std, test_B_peak_to_peak, test_B_skew, test_B_kurtosis))
X_test_predict = np.hstack((X_test_predict, test_material_encoded.values, test_waveform_encoded.values))

# 进行预测
test_predictions = rf_model.predict(X_test_predict)

# 保留一位小数
test_predictions_rounded = np.round(test_predictions, 1)

# 将结果写入附件四
file_output_path = '附件四.xlsx'
output_data = test_data.copy()
output_data['磁芯损耗预测'] = test_predictions_rounded
output_data.to_excel(file_output_path, index=False)

# 特别输出指定样本序号的预测结果
special_indices = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
special_samples = output_data.iloc[special_indices]
print("\n指定样本序号的磁芯损耗预测结果：")
print(special_samples[['样本序号', '磁芯损耗预测']])
