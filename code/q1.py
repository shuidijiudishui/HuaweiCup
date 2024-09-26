import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import scipy.stats as stats

# Step 1: Load the Excel data
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')
material_2 = pd.read_excel(file_path, sheet_name='材料2')
material_3 = pd.read_excel(file_path, sheet_name='材料3')
material_4 = pd.read_excel(file_path, sheet_name='材料4')


# Step 2: Function to extract statistical features from flux density data
def extract_statistical_features(flux_density_data):
    flux_density_data = flux_density_data.astype(float)

    # 统计特征
    B_mean = np.mean(flux_density_data)  # 磁通密度均值
    B_std = np.std(flux_density_data)  # 磁通密度标准差
    B_peak_to_peak = np.max(flux_density_data) - np.min(flux_density_data)  # 峰峰值
    B_skew = stats.skew(flux_density_data)  # 偏度
    B_kurtosis = stats.kurtosis(flux_density_data)  # 峰度

    # 傅里叶特征
    fft_vals = np.fft.fft(flux_density_data)
    fft_freq = np.fft.fftfreq(len(flux_density_data))

    # 提取前三个傅里叶频率的幅值
    fft_amplitudes = np.abs(fft_vals)
    top3_fft_features = np.sort(fft_amplitudes)[-3:]  # 选择前三大幅值

    return [B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis] + list(top3_fft_features)


# Step 3: Create feature matrix from the materials data
def create_feature_matrix(material_data):
    features = []
    labels = material_data.iloc[:, 3].values  # Column 3 contains waveform types
    for i in range(material_data.shape[0]):
        flux_density = material_data.iloc[i, 4:].values  # Extract flux density data from column 5 onwards
        stat_features = extract_statistical_features(flux_density)
        features.append(stat_features)
    return np.array(features), labels


# Extract features from each material
X1, y1 = create_feature_matrix(material_1)
X2, y2 = create_feature_matrix(material_2)
X3, y3 = create_feature_matrix(material_3)
X4, y4 = create_feature_matrix(material_4)

# Combine all the data
X = np.vstack((X1, X2, X3, X4))
y = np.hstack((y1, y2, y3, y4))

# Step 4: PCA for dimensionality reduction
pca = PCA(n_components=3)  # Reduce to 3 dimensions
X_pca = pca.fit_transform(X)

# Step 5: Build and train the classification model
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=41)
clf = RandomForestClassifier(n_estimators=200, random_state=41)
clf.fit(X_train, y_train)

# Evaluate the model on the training dataset
y_pred_train = clf.predict(X_test)
print(classification_report(y_test, y_pred_train))

# Step 6: Predict waveform types in 附件二（测试集）
test_data_2 = pd.read_excel('附件二（测试集）.xlsx')


def extract_test_features(test_data):
    features = []
    for i in range(test_data.shape[0]):
        flux_density = test_data.iloc[i, 5:].values  # Extract flux density data from column 5 onwards
        flux_density = flux_density.astype(float)  # Ensure the data is in float format
        stat_features = extract_statistical_features(flux_density)
        features.append(stat_features)
    return np.array(features)


X_test_features_2 = extract_test_features(test_data_2)
X_test_features_2_pca = pca.transform(X_test_features_2)
y_test_pred_2 = clf.predict(X_test_features_2_pca)

# Step 7: Save the results to '附件四.xlsx'
test_data_2['波形类型'] = y_test_pred_2
test_data_2['波形类型'] = test_data_2['波形类型'].map({'正弦波': 1, '三角波': 2, '梯形波': 3})
test_data_2.to_excel('附件四.xlsx', index=False)

# Step 8: Predict on 附件三（测试集）
# test_data_3 = pd.read_excel('附件二（测试集）.xlsx')
# X_test_features_3 = extract_test_features(test_data_3)
# X_test_features_3_pca = pca.transform(X_test_features_3)
#
# if '励磁波形' in test_data_3.columns:
#     y_true_3 = test_data_3['励磁波形'].values
#     y_test_pred_3 = clf.predict(X_test_features_3_pca)
#     from sklearn.metrics import accuracy_score
#
#     accuracy_3 = accuracy_score(y_true_3, y_test_pred_3)
#     print(f'Accuracy on 附件三: {accuracy_3 * 100:.2f}%')
#
#     misclassified_indices = np.where(y_test_pred_3 != y_true_3)[0]  # 找到预测错误的索引
#     if len(misclassified_indices) > 0:
#         print(f"Number of misclassified samples: {len(misclassified_indices)}")
#         print("Details of misclassified samples:")
#         misclassified_samples = test_data_3.iloc[misclassified_indices]
#         print(misclassified_samples)
#         misclassified_samples.to_excel('预测错误的样本_附件三.xlsx', index=False)
#     else:
#         print("All samples were classified correctly!")
