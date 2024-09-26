import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取数据
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']


def steinmetz_eq(x, k1, alpha1, beta1):
    f, B_m = x
    return k1 * (f ** alpha1) * (B_m ** beta1)


# 预处理数据，计算磁通密度的峰值
def preprocess_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    # 计算磁通密度峰值
    B_m = data.iloc[:, 4:].max(axis=1)
    # 筛选出正弦波形的数据
    sine_wave_data = data[data.iloc[:, 3] == '正弦波']
    return sine_wave_data.iloc[:, [0, 1, 2]], B_m


# 对每个温度进行拟合，并计算评价指标
def fit_steinmetz_equation(data, B_m):
    results = {}
    temperatures = data.iloc[:, 0].unique()
    MAEs = []
    MSEs = []
    R2s = []

    for temp in temperatures:
        temp_data = data[data.iloc[:, 0] == temp]
        f_values = temp_data.iloc[:, 1].values  # 频率
        P_values = temp_data.iloc[:, 2].values  # 磁芯损耗
        B_m_values = B_m[temp_data.index]  # 磁通密度峰值

        # 拟合斯坦麦茨方程
        popt, pcov = curve_fit(steinmetz_eq, (f_values, B_m_values), P_values,
                               p0=(1e-5, 2, 2))  # 初始猜测的参数
        k1, alpha1, beta1 = popt
        results[temp] = {'k1': k1, 'alpha1': alpha1, 'beta1': beta1}

        # 调用斯坦麦茨方程进行预测
        P_pred = steinmetz_eq((f_values, B_m_values), *popt)

        # 计算 MAE, MSE, R²
        MAEs.append(mean_absolute_error(P_values, P_pred))
        MSEs.append(mean_squared_error(P_values, P_pred))
        R2s.append(r2_score(P_values, P_pred))

        # 输出拟合结果
        print(f"温度 {temp}°C 下的拟合参数：")
        print(f"k1 = {k1}, alpha1 = {alpha1}, beta1 = {beta1}")
        print(f"MAE = {MAEs[-1]}, MSE = {MSEs[-1]}, R² = {R2s[-1]}")

    # 绘制 MAE, MSE 和 R² 的图，3x1 布局
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    fig, axes = plt.subplots(3, 1, figsize=(4, 6))

    # MAE 图
    axes[0].plot(temperatures, MAEs, 'o-', color='blue')
    axes[0].set_title('MAE 随温度变化图', fontsize=10)
    axes[0].set_xlabel('温度 (°C)')
    axes[0].set_ylabel('MAE')

    # MSE 图
    axes[1].plot(temperatures, MSEs, 'o-', color='green')
    axes[1].set_title('MSE 随温度变化图', fontsize=10)
    axes[1].set_xlabel('温度 (°C)')
    axes[1].set_ylabel('MSE')

    # R² 图
    axes[2].plot(temperatures, R2s, 'o-', color='red')
    axes[2].set_title('R² 随温度变化图', fontsize=10)
    axes[2].set_xlabel('温度 (°C)')
    axes[2].set_ylabel('R²')

    plt.tight_layout()
    plt.show()

    return results


# 处理材料1数据并进行拟合
sine_wave_data, B_m_values = preprocess_data('材料1')
fit_results = fit_steinmetz_equation(sine_wave_data, B_m_values)
