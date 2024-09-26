import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

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

# Step 4: Mutual Information Analysis
mi_scores = mutual_info_classif(X, y, random_state=41)
mi_scores = pd.Series(mi_scores, index=['均值', '标准差', '峰峰值', '偏度', '峰度',
                                        '一次谐波幅值', '二次谐波幅值', '三次谐波幅值'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
# Visualize Mutual Information Scores
plt.figure(figsize=(8, 6))
mi_scores.sort_values(ascending=False).plot(kind='bar')
plt.title('各特征对励磁波形的互信息分数')
plt.ylabel('分数')
plt.xticks(rotation=45, ha='right', fontsize=8)  # 设置x轴刻度字体大小，并倾斜45度plt.yticks(fontsize=6)  # 设置y轴刻度字体大小
plt.show()

# Step 5: PCA for dimensionality reduction
pca = PCA(n_components=3)  # Reduce to 3 dimensions for 3D visualization
X_pca = pca.fit_transform(X)

# Visualize the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors for different waveform types
colors = {1: 'r', 2: 'g', 3: 'b'}
waveform_labels = [1 if val == '正弦波' else 2 if val == '三角波' else 3 for val in y]

# Create 3D scatter plot
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=[colors[label] for label in waveform_labels], s=50)

# Add labels and title
ax.set_xlabel('主成分1')
ax.set_ylabel('主成分2')
ax.set_zlabel('主成分3')
plt.title('主成分分析')
plt.show()

# Step 6: Build and train the classification model using PCA-transformed features
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=41)
clf = RandomForestClassifier(n_estimators=200, random_state=41)
clf.fit(X_train, y_train)

# Evaluate the model on the test dataset
y_pred_train = clf.predict(X_test)
print(classification_report(y_test, y_pred_train))
