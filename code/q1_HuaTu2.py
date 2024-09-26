import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. 加载数据
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')


# 2. 分布特征
def calculate_features(flux_density_data):
    flux_density_data = flux_density_data.astype(float)
    B_mean = np.mean(flux_density_data)  # 均值
    B_std = np.std(flux_density_data)  # 标准差
    B_peak_to_peak = np.max(flux_density_data) - np.min(flux_density_data)  # 峰峰值
    B_skewness = stats.skew(flux_density_data)  # 偏度
    B_kurtosis = stats.kurtosis(flux_density_data)  # 峰度
    return B_mean, B_std, B_peak_to_peak, B_skewness, B_kurtosis


# Step 3: Create lists to store feature values
mean_values = []
std_values = []
peak_to_peak_values = []
skewness_values = []
kurtosis_values = []

# Step 4: Calculate features for each row (each sample)
for i in range(material_1.shape[0]):
    flux_density = material_1.iloc[i, 4:].values  # 从第5列开始是磁通密度数据
    B_mean, B_std, B_peak_to_peak, B_skewness, B_kurtosis = calculate_features(flux_density)

    # Append the results to the corresponding lists
    mean_values.append(B_mean)
    std_values.append(B_std)
    peak_to_peak_values.append(B_peak_to_peak)
    skewness_values.append(B_skewness)
    kurtosis_values.append(B_kurtosis)

# Step 5: Plot the features against sample index
sample_indices = np.arange(material_1.shape[0])
plt.figure(figsize=(10, 8))

# 设置全局字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# Plot for mean values
plt.subplot(2, 3, 1)
plt.plot(sample_indices, mean_values, label='均值')
plt.xlabel('样本号')
plt.ylabel('均值 (T)')
plt.title('每个样本的磁通密度均值', fontsize=10)  # 设置标题字体大小为 10

# Plot for standard deviation values
plt.subplot(2, 3, 2)
plt.plot(sample_indices, std_values, label='标准差', color='orange')
plt.xlabel('样本号')
plt.ylabel('标准差 (T)')
plt.title('每个样本的磁通密度标准差', fontsize=10)  # 设置标题字体大小为 10

# Plot for peak-to-peak values
plt.subplot(2, 3, 3)
plt.plot(sample_indices, peak_to_peak_values, label='峰峰值', color='green')
plt.xlabel('样本号')
plt.ylabel('峰峰值 (T)')
plt.title('每个样本的峰峰值', fontsize=10)  # 设置标题字体大小为 10

# Plot for skewness values
plt.subplot(2, 3, 4)
plt.plot(sample_indices, skewness_values, label='偏度', color='red')
plt.xlabel('样本号')
plt.ylabel('偏度')
plt.title('每个样本的偏度', fontsize=10)  # 设置标题字体大小为 10

# Plot for kurtosis values
plt.subplot(2, 3, 5)
plt.plot(sample_indices, kurtosis_values, label='峰度', color='purple')
plt.xlabel('样本号')
plt.ylabel('峰度')
plt.title('每个样本的峰度', fontsize=10)  # 设置标题字体大小为 10

# 调整子图间距
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.tight_layout()
plt.show()
