import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图形绘制

# 读取数据
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']

# 斯坦麦茨方程
def steinmetz_eq(x, k1, alpha1, beta1):
    f, B_m = x
    return k1 * (f**alpha1) * (B_m**beta1)

# 预处理数据，计算磁通密度的峰值
def preprocess_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    # 计算磁通密度峰值
    B_m = data.iloc[:, 4:].max(axis=1)
    # 筛选出正弦波形的数据
    sine_wave_data = data[data.iloc[:, 3] == '正弦波']
    return sine_wave_data.iloc[:, [0, 1, 2]], B_m

# 对每个温度进行拟合并绘制三维图
def fit_steinmetz_equation(data, B_m):
    results = {}
    temperatures = data.iloc[:, 0].unique()

    # 创建一个 2x2 的布局
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    fig = plt.figure(figsize=(12, 8))

    for i, temp in enumerate(temperatures):
        temp_data = data[data.iloc[:, 0] == temp]
        f_values = temp_data.iloc[:, 1].values  # 频率
        P_values = temp_data.iloc[:, 2].values  # 磁芯损耗
        B_m_values = B_m[temp_data.index]  # 磁通密度峰值

        # 拟合斯坦麦茨方程
        popt, pcov = curve_fit(steinmetz_eq, (f_values, B_m_values), P_values,
                               p0=(1e-5, 2, 2))  # 初始猜测的参数
        k1, alpha1, beta1 = popt
        results[temp] = {'k1': k1, 'alpha1': alpha1, 'beta1': beta1}

        # 创建子图，并绘制三维拟合曲线
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(f_values, B_m_values, P_values, label='实际数据')

        # 创建频率和磁通密度网格，用于绘制拟合曲线
        f_grid, B_m_grid = np.meshgrid(np.logspace(np.log10(f_values.min()), np.log10(f_values.max()), 100),
                                       np.linspace(B_m_values.min(), B_m_values.max(), 100))
        P_pred_grid = steinmetz_eq((f_grid, B_m_grid), *popt)

        # 绘制拟合曲面
        ax.plot_surface(f_grid, B_m_grid, P_pred_grid, color='red', alpha=0.5)

        # 设置三维图形的轴标签
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('最大磁通密度 (T)')
        ax.set_zlabel('磁芯损耗')
        ax.set_title(f'温度 {temp}°C 下的斯坦麦茨方程拟合', fontsize=10)

    # 调整子图布局
    plt.tight_layout()
    plt.show()

    return results

# 处理材料1数据并进行拟合
sine_wave_data, B_m_values = preprocess_data('材料1')
fit_results = fit_steinmetz_equation(sine_wave_data, B_m_values)
