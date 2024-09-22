import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: 读取数据
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']
all_data = []

for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['材料'] = sheet  # 添加“材料”列
    all_data.append(data)

all_materials_data = pd.concat(all_data, ignore_index=True)


# Step 2: 定义特征提取函数
def extract_waveform_features(B_data):
    """
    提取磁通密度曲线的形状特征：
    - 峰值数量
    - 谷值数量
    - 斜率突变点数量
    """
    features = []
    for i in range(B_data.shape[0]):
        # 获取当前样本的磁通密度序列
        B_sample = B_data.iloc[i, :].values

        # 计算峰值和谷值数量
        peaks, _ = find_peaks(B_sample)
        troughs, _ = find_peaks(-B_sample)
        num_peaks = len(peaks)
        num_troughs = len(troughs)

        # 计算斜率突变点数量
        slopes = np.diff(B_sample)  # 一阶导数
        slope_changes = np.where(np.diff(np.sign(slopes)))[0]  # 斜率符号变化点
        num_slope_changes = len(slope_changes)

        # 记录特征
        features.append([num_peaks, num_troughs, num_slope_changes])

    return np.array(features)


# Step 3: 提取磁通密度数据并计算特征
B_columns = all_materials_data.iloc[:, 4:1029]  # 磁通密度数据
features = extract_waveform_features(B_columns)

# Step 4: 将励磁波形类型（正弦波、三角波、梯形波）转换为数值
waveform_map = {'正弦波': 1, '三角波': 2, '梯形波': 3}
waveform_labels = all_materials_data.iloc[:, 3].map(waveform_map).values

# Step 5: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, waveform_labels, test_size=0.2, random_state=42)

# Step 6: 建立分类模型（随机森林分类器）
clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

# Step 7: 评估模型
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['正弦波', '三角波', '梯形波']))

# Step 8: 应用到附件二的样本上
file_test_path = '附件二（测试集）.xlsx'
test_data = pd.read_excel(file_test_path)

# 提取测试集的磁通密度数据
test_B_columns = test_data.iloc[:, 4:1029]

# 提取特征
test_features = extract_waveform_features(test_B_columns)

# 使用模型进行预测
test_waveform_predictions = clf.predict(test_features)

# Step 9: 将结果填入附件四
file_output_path = '附件四.xlsx'
output_data = test_data.copy()
output_data['励磁波形分类'] = test_waveform_predictions  # 添加分类结果

# 保存结果
output_data.to_excel(file_output_path, index=False)

# Step 10: 输出指定样本序号的分类结果
special_indices = [0, 4, 14, 24, 34, 44, 54, 64, 74, 79]  # 注意样本序号从0开始
special_samples = output_data.iloc[special_indices]
print("\n指定样本序号的分类结果：")
print(special_samples[['样本序号', '励磁波形分类']])

# Step 11: 统计各类波形的数量
waveform_counts = output_data['励磁波形分类'].value_counts()
print("\n三种波形的数量统计：")
print(waveform_counts)
