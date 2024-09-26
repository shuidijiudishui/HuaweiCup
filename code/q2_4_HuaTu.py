import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
file_path = '附件一（训练集）.xlsx'

def steinmetz_eq(x, k1, alpha1, beta1):
    f, B_m = x.T
    return k1 * (f ** alpha1) * (B_m ** beta1)

# 定义修正方程
def linear_correction(x, a, alpha1, beta1):
    T, f, B_m = x.T
    return a * T * (f ** alpha1) * (B_m ** beta1)

def quadratic_correction(x, a, b, c, alpha1, beta1):
    T, f, B_m = x.T
    return (a * T ** 2 + b * T + c) * (f ** alpha1) * (B_m ** beta1)

def exponential_correction(x, a, b, alpha1, beta1):
    T, f, B_m = x.T
    return a * np.exp(b * T) * (f ** alpha1) * (B_m ** beta1)

def preprocess_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    B_m = data.iloc[:, 4:].max(axis=1)
    sine_wave_data = data[data.iloc[:, 3] == '正弦波']
    return sine_wave_data.iloc[:, [0, 1, 2]], B_m[sine_wave_data.index]

def fit_and_compare_models(data, B_m):
    temperatures = data.iloc[:, 0].values
    f_values = data.iloc[:, 1].values
    P_values = data.iloc[:, 2].values
    B_m_values = B_m.values

    x_data_steinmetz = np.column_stack((f_values, B_m_values))
    popt_steinmetz, _ = curve_fit(steinmetz_eq, x_data_steinmetz, P_values, p0=(1e-5, 2, 2))

    x_data_all = np.column_stack((temperatures, f_values, B_m_values))
    maxfev_value = 100000
    popt_linear, _ = curve_fit(linear_correction, x_data_all, P_values, p0=(1e-5, 2, 2), maxfev=maxfev_value)
    popt_quadratic, _ = curve_fit(quadratic_correction, x_data_all, P_values, p0=(1e-5, 1e-5, 1, 2, 2), maxfev=maxfev_value)
    popt_exponential, _ = curve_fit(exponential_correction, x_data_all, P_values, p0=(1e-5, 1e-5, 2, 2), maxfev=maxfev_value)

    metrics = {'MAE': [], 'MSE': [], 'R2': []}
    unique_temperatures = np.unique(temperatures)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    for temp in unique_temperatures:
        mask = temperatures == temp
        f_values_temp = f_values[mask]
        P_values_temp = P_values[mask]
        B_m_values_temp = B_m_values[mask]

        sorted_indices = np.argsort(f_values_temp)
        f_values_sorted = f_values_temp[sorted_indices]
        P_values_sorted = P_values_temp[sorted_indices]

        # 计算预测值
        P_pred_steinmetz_sorted = steinmetz_eq(x_data_steinmetz[mask], *popt_steinmetz)[sorted_indices]
        P_pred_linear_sorted = linear_correction(x_data_all[mask], *popt_linear)[sorted_indices]
        P_pred_quadratic_sorted = quadratic_correction(x_data_all[mask], *popt_quadratic)[sorted_indices]
        P_pred_exponential_sorted = exponential_correction(x_data_all[mask], *popt_exponential)[sorted_indices]

        # 计算指标
        metrics['MAE'].append([
            mean_absolute_error(P_values_sorted, P_pred_steinmetz_sorted),
            mean_absolute_error(P_values_sorted, P_pred_linear_sorted),
            mean_absolute_error(P_values_sorted, P_pred_quadratic_sorted),
            mean_absolute_error(P_values_sorted, P_pred_exponential_sorted)
        ])

        metrics['MSE'].append([
            mean_squared_error(P_values_sorted, P_pred_steinmetz_sorted),
            mean_squared_error(P_values_sorted, P_pred_linear_sorted),
            mean_squared_error(P_values_sorted, P_pred_quadratic_sorted),
            mean_squared_error(P_values_sorted, P_pred_exponential_sorted)
        ])

        metrics['R2'].append([
            r2_score(P_values_sorted, P_pred_steinmetz_sorted),
            r2_score(P_values_sorted, P_pred_linear_sorted),
            r2_score(P_values_sorted, P_pred_quadratic_sorted),
            r2_score(P_values_sorted, P_pred_exponential_sorted)
        ])

    return metrics, unique_temperatures

# 处理材料1数据并进行拟合比较
sine_wave_data, B_m_values = preprocess_data('材料1')
metrics, unique_temperatures = fit_and_compare_models(sine_wave_data, B_m_values)

# 绘制直方图
model_names = ['斯坦麦茨方程', '线性修正斯坦麦茨方程', '二次修正斯坦麦茨方程', '指数修正斯坦麦茨方程']
metrics_names = ['MAE', 'MSE', 'R2']

n_metrics = len(metrics_names)
n_models = len(model_names)

# 动态创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics_names):
    for j, temp in enumerate(unique_temperatures):
        axes[i].bar(np.arange(n_models) + j * 0.2, metrics[metric][j], width=0.2, label=f'{temp}°C')

    axes[i].set_xticks(np.arange(n_models) + 0.2)
    axes[i].set_xticklabels(model_names)
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'不同温度下的{metric}比较')
    axes[i].legend(title='温度')

# 移除多余的子图
if n_metrics < 4:
    for k in range(n_metrics, 4):
        fig.delaxes(axes[k])

plt.tight_layout()
plt.show()