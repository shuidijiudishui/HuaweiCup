import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Excel data for all materials
file_path = '附件一（训练集）.xlsx'
materials = {
    '材料1': pd.read_excel(file_path, sheet_name='材料1'),
    '材料2': pd.read_excel(file_path, sheet_name='材料2'),
    '材料3': pd.read_excel(file_path, sheet_name='材料3'),
    '材料4': pd.read_excel(file_path, sheet_name='材料4')
}

# Step 2: Function to randomly select samples with a specific waveform
def select_random_samples(material_data, waveform_type, num_samples=10):
    waveform_samples = material_data[material_data.iloc[:, 3] == waveform_type]  # 筛选出指定波形类型的样本
    return waveform_samples.sample(n=num_samples, random_state=42)  # 随机选择指定数量的样本

# Step 3: Function to plot the flux density for selected samples on a given axis
def plot_flux_density(ax, waveform_type):
    time = np.arange(1024)  # 时间轴
    # 设置字体为微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    # Loop through each material
    for material_name, material_data in materials.items():
        # Randomly select 10 samples for the specified waveform type
        selected_samples = select_random_samples(material_data, waveform_type, num_samples=10)

        # Plot each sample's flux density
        for idx in range(selected_samples.shape[0]):
            sample_flux_density = selected_samples.iloc[idx, 4:].values  # 提取磁通密度数据
            ax.plot(time, sample_flux_density, label=f'{material_name} - 样本序号 {selected_samples.index[idx]}')

    # 设置图表样式
    ax.set_title(f'{waveform_type}波形', fontsize=10)
    ax.set_ylabel('磁通密度 (T)', fontsize=8)
    ax.grid(True)
    ax.legend(fontsize=5)

# Step 4: Create subplots for each waveform type
fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # 创建1x3的子图布局


# 正弦波
plot_flux_density(axs[0], '正弦波')

# 三角波
plot_flux_density(axs[1], '三角波')

# 梯形波
plot_flux_density(axs[2], '梯形波')

# 全局设置
plt.suptitle('不同波形下的磁通密度变化曲线（每种材料随机选取10个样本）', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以避免重叠
plt.show()
