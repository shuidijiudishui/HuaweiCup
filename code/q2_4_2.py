import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 读取数据
file_path = '附件一（训练集）.xlsx'

def steinmetz_eq(x, k1, alpha1, beta1):
    f, B_m = x.T
    return k1 * (f ** alpha1) * (B_m ** beta1)

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

def fit_and_output_coefficients(data, B_m):
    temperatures = data.iloc[:, 0].values
    f_values = data.iloc[:, 1].values
    P_values = data.iloc[:, 2].values
    B_m_values = B_m.values

    unique_temperatures = np.unique(temperatures)

    for temp in unique_temperatures:
        mask = temperatures == temp
        f_values_temp = f_values[mask]
        P_values_temp = P_values[mask]
        B_m_values_temp = B_m_values[mask]

        x_data_steinmetz = np.column_stack((f_values_temp, B_m_values_temp))
        popt_steinmetz, _ = curve_fit(steinmetz_eq, x_data_steinmetz, P_values_temp, p0=(1e-5, 2, 2))

        x_data_all = np.column_stack((np.full_like(f_values_temp, temp), f_values_temp, B_m_values_temp))
        popt_linear, _ = curve_fit(linear_correction, x_data_all, P_values_temp, p0=(1e-5, 2, 2))
        popt_quadratic, _ = curve_fit(quadratic_correction, x_data_all, P_values_temp, p0=(1e-5, 1e-5, 1, 2, 2))
        popt_exponential, _ = curve_fit(exponential_correction, x_data_all, P_values_temp, p0=(1e-5, 1e-5, 2, 2))

        print(f"温度: {temp}°C")
        print(f"斯坦麦茨方程拟合系数: {popt_steinmetz}")
        print(f"线性修正拟合系数: {popt_linear}")
        print(f"二次修正拟合系数: {popt_quadratic}")
        print(f"指数修正拟合系数: {popt_exponential}\n")

# 处理材料1数据并进行拟合输出
sine_wave_data, B_m_values = preprocess_data('材料1')
fit_and_output_coefficients(sine_wave_data, B_m_values)
