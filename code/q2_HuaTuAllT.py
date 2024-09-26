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


# 对所有温度进行整体拟合，并计算评价指标
def fit_steinmetz_equation(data, B_m):
    f_values = data.iloc[:, 1].values  # 频率
    P_values = data.iloc[:, 2].values  # 磁芯损耗
    B_m_values = B_m[data.index]  # 磁通密度峰值

    # 拟合斯坦麦茨方程
    popt, pcov = curve_fit(steinmetz_eq, (f_values, B_m_values), P_values, p0=(1e-5, 2, 2))  # 初始猜测的参数
    k1, alpha1, beta1 = popt

    # 调用斯坦麦茨方程进行预测
    P_pred = steinmetz_eq((f_values, B_m_values), *popt)

    # 计算 MAE, MSE, R²
    MAE = mean_absolute_error(P_values, P_pred)
    MSE = mean_squared_error(P_values, P_pred)
    R2 = r2_score(P_values, P_pred)

    # 输出拟合结果
    print(f"整体拟合参数：")
    print(f"k1 = {k1}, alpha1 = {alpha1}, beta1 = {beta1}")
    print(f"MAE = {MAE}, MSE = {MSE}, R² = {R2}")

    # 绘制 MAE, MSE, R² 的直方图
    # 设置全局字体为微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    metrics = {'MAE': MAE, 'MSE': MSE, 'R²': R2}
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        axes[i].bar([metric_name], [metric_value], color='skyblue')
        axes[i].set_title(f'{metric_name} 直方图')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, max(metrics.values()) * 1.2)  # 设置y轴范围

    plt.tight_layout()
    plt.show()

    return popt


# 处理材料1数据并进行拟合
sine_wave_data, B_m_values = preprocess_data('材料1')
fit_results = fit_steinmetz_equation(sine_wave_data, B_m_values)
