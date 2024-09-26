import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# 读取数据
file_path = '附件一（训练集）.xlsx'


def steinmetz_eq(x, k1, alpha1, beta1):
    f, B_m = x.T  # 转置以解包数据
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

# 预处理数据，计算磁通密度的峰值
def preprocess_data(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    # 计算磁通密度峰值
    B_m = data.iloc[:, 4:].max(axis=1)
    # 筛选出正弦波形的数据
    sine_wave_data = data[data.iloc[:, 3] == '正弦波']
    return sine_wave_data.iloc[:, [0, 1, 2]], B_m[sine_wave_data.index]

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def fit_and_compare_models(data, B_m):
    temperatures = data.iloc[:, 0].values  # 温度
    f_values = data.iloc[:, 1].values  # 频率
    P_values = data.iloc[:, 2].values  # 磁芯损耗
    B_m_values = B_m.values  # 磁通密度峰值

    # 将 f_values 和 B_m_values 列为二维数组
    x_data_steinmetz = np.column_stack((f_values, B_m_values))

    # 斯坦麦茨方程拟合
    popt_steinmetz, _ = curve_fit(steinmetz_eq, x_data_steinmetz, P_values, p0=(1e-5, 2, 2))

    # 将温度、频率和磁通密度列为二维数组
    x_data_all = np.column_stack((temperatures, f_values, B_m_values))

    # 增加 maxfev 来提升最大迭代次数
    maxfev_value = 100000

    # 修正方程拟合
    popt_linear, _ = curve_fit(linear_correction, x_data_all, P_values, p0=(1e-5, 2, 2), maxfev=maxfev_value)
    popt_quadratic, _ = curve_fit(quadratic_correction, x_data_all, P_values, p0=(1e-5, 1e-5, 1, 2, 2),
                                  maxfev=maxfev_value)
    popt_exponential, _ = curve_fit(exponential_correction, x_data_all, P_values, p0=(1e-5, 1e-5, 2, 2),
                                    maxfev=maxfev_value)

    # 预测值计算
    P_pred_steinmetz = steinmetz_eq(x_data_steinmetz, *popt_steinmetz)
    P_pred_linear = linear_correction(x_data_all, *popt_linear)
    P_pred_quadratic = quadratic_correction(x_data_all, *popt_quadratic)
    P_pred_exponential = exponential_correction(x_data_all, *popt_exponential)

    # 设置全局字体为微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 存储不同温度下的MAE、MSE、R²值
    mae_results = {}
    mse_results = {}
    r2_results = {}

    # 不同温度下分别计算MAE、MSE和R²
    unique_temperatures = np.unique(temperatures)

    # 初始化图形
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 用于跟踪子图的位置
    subplot_idx = 0

    for temp in unique_temperatures:
        mask = temperatures == temp
        f_values_temp = f_values[mask]
        P_values_temp = P_values[mask]
        B_m_values_temp = B_m_values[mask]

        # 对频率进行排序
        sorted_indices = np.argsort(f_values_temp)
        f_values_sorted = f_values_temp[sorted_indices]
        P_values_sorted = P_values_temp[sorted_indices]

        # 对预测值也进行排序
        P_pred_steinmetz_sorted = P_pred_steinmetz[mask][sorted_indices]
        P_pred_linear_sorted = P_pred_linear[mask][sorted_indices]
        P_pred_quadratic_sorted = P_pred_quadratic[mask][sorted_indices]
        P_pred_exponential_sorted = P_pred_exponential[mask][sorted_indices]

        # 计算每种模型在该温度下的 MAE、MSE 和 R²
        mse_steinmetz = mean_squared_error(P_values_sorted, P_pred_steinmetz_sorted)
        mae_steinmetz = mean_absolute_error(P_values_sorted, P_pred_steinmetz_sorted)
        r2_steinmetz = r2_score(P_values_sorted, P_pred_steinmetz_sorted)

        mse_linear = mean_squared_error(P_values_sorted, P_pred_linear_sorted)
        mae_linear = mean_absolute_error(P_values_sorted, P_pred_linear_sorted)
        r2_linear = r2_score(P_values_sorted, P_pred_linear_sorted)

        mse_quadratic = mean_squared_error(P_values_sorted, P_pred_quadratic_sorted)
        mae_quadratic = mean_absolute_error(P_values_sorted, P_pred_quadratic_sorted)
        r2_quadratic = r2_score(P_values_sorted, P_pred_quadratic_sorted)

        mse_exponential = mean_squared_error(P_values_sorted, P_pred_exponential_sorted)
        mae_exponential = mean_absolute_error(P_values_sorted, P_pred_exponential_sorted)
        r2_exponential = r2_score(P_values_sorted, P_pred_exponential_sorted)

        print(f"温度: {temp}°C")
        print(f"斯坦麦茨方程拟合系数: {popt_steinmetz}")
        print(f"斯坦麦茨方程: MSE={mse_steinmetz}, MAE={mae_steinmetz}, R²={r2_steinmetz}")
        print(f"线性修正拟合系数: {popt_linear}")
        print(f"线性修正: MSE={mse_linear}, MAE={mae_linear}, R²={r2_linear}")
        print(f"二次修正拟合系数: {popt_quadratic}")
        print(f"二次修正: MSE={mse_quadratic}, MAE={mae_quadratic}, R²={r2_quadratic}")
        print(f"指数修正拟合系数: {popt_exponential}")
        print(f"指数修正: MSE={mse_exponential}, MAE={mae_exponential}, R²={r2_exponential}")

        ax = axs[subplot_idx // 2, subplot_idx % 2]
        ax.scatter(f_values_sorted, P_values_sorted, label=f'实际值 {temp}°C', color='black', alpha=0.7)
        ax.plot(f_values_sorted, P_pred_steinmetz_sorted, label='斯坦麦茨方程', linestyle='--', alpha=0.7)
        ax.plot(f_values_sorted, P_pred_linear_sorted, label='线性修正', linestyle='-.', alpha=0.7)
        ax.plot(f_values_sorted, P_pred_quadratic_sorted, label='二次修正', linestyle=':', alpha=0.7)
        ax.plot(f_values_sorted, P_pred_exponential_sorted, label='指数修正', linestyle='-', alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('磁芯损耗 (W/m³)')
        ax.set_title(f'{temp}°C 温度下不同修正方程的拟合效果比较',fontsize=10)
        ax.legend()

        subplot_idx += 1

        if subplot_idx >= 4:  # 如果已经绘制了四个子图，则停止
            break

    plt.tight_layout()
    plt.show()


# 处理材料1数据并进行拟合比较
sine_wave_data, B_m_values = preprocess_data('材料1')
fit_and_compare_models(sine_wave_data, B_m_values)
