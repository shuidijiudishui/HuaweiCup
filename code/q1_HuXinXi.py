import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Step 1: 读取数据
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']

# 初始化列表以存储所有数据
all_data = []

# 遍历每个 sheet
for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['材料'] = sheet  # 添加“材料”列
    all_data.append(data)

# 将所有材料的数据合并为一个 DataFrame
all_materials_data = pd.concat(all_data, ignore_index=True)


# Step 2: 计算形状特征
def calculate_shape_features(B_columns):
    features = {}
    features['B_mean'] = np.mean(B_columns, axis=1)
    features['B_std'] = np.std(B_columns, axis=1)
    features['B_peak_to_peak'] = np.ptp(B_columns, axis=1)
    features['B_skew'] = skew(B_columns, axis=1)
    features['B_kurtosis'] = kurtosis(B_columns, axis=1)

    # 傅里叶变换提取前几个主要频率成分
    fft_values = fft(B_columns, axis=1)
    features['fft_1'] = np.abs(fft_values[:, 1])
    features['fft_2'] = np.abs(fft_values[:, 2])
    features['fft_3'] = np.abs(fft_values[:, 3])

    return pd.DataFrame(features)


# 提取磁通密度列
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算形状特征
shape_features = calculate_shape_features(B_columns)

# Step 3: 将励磁波形编码为标签
waveform = all_materials_data['励磁波形']
le = LabelEncoder()
waveform_labels = le.fit_transform(waveform)

# Step 4: 数据合并
data_with_features = pd.concat([shape_features, pd.Series(waveform_labels, name='waveform_label')], axis=1)

# Step 5: 划分训练集和测试集
X = data_with_features.drop(columns='waveform_label')
y = data_with_features['waveform_label']

imputer = SimpleImputer(strategy='mean')  # 你也可以选择 'median' 或其他策略
X_imputed = imputer.fit_transform(X)

# Step 1: 计算互信息
mi = mutual_info_classif(X, y, random_state=42)

# Step 2: 将结果存储为 DataFrame 方便查看
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual Information': mi
}).sort_values(by='Mutual Information', ascending=False)

# 输出互信息结果
print(mi_df)

# Step 3: 可视化互信息结果
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'], mi_df['Mutual Information'], color='skyblue')
plt.xlabel('Mutual Information')
plt.ylabel('Features')
plt.title('Mutual Information between Features and Waveform Types')
plt.gca().invert_yaxis()
plt.show()
