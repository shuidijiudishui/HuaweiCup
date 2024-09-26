import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: 读取材料1的正弦波数据
file_path = '附件一（训练集）.xlsx'
material_1 = pd.read_excel(file_path, sheet_name='材料1')

# 筛选出正弦波形的数据
sine_wave_data = material_1[material_1['励磁波形'] == '正弦波']


# Step 2: 定义斯坦麦茨方程和修正方程
def steinmetz(frequency, k, alpha, beta):
    return k * frequency ** alpha


def corrected_steinmetz(frequency, temperature, k, alpha, beta, n):
    return k * frequency ** alpha * temperature ** n


# Step 3: 对材料1的不同温度下的频率和损耗进行拟合
temperatures = sine_wave_data['温度，oC'].unique()

for temp in temperatures:
    data_at_temp = sine_wave_data[sine_wave_data['温度，oC'] == temp]
    frequency = data_at_temp['频率，Hz'].values
    core_loss = data_at_temp['磁芯损耗，w/m3'].values

    # 拟合斯坦麦茨方程
    popt_steinmetz, _ = curve_fit(steinmetz, frequency, core_loss)

    # 绘制频率与损耗的散点图
    plt.scatter(frequency, core_loss, label=f'T={temp}°C')

    # 预测损耗值
    predicted_loss = steinmetz(frequency, *popt_steinmetz)
    plt.plot(frequency, predicted_loss, label=f'Predicted T={temp}°C')

# 绘制图形
plt.xlabel('频率 (Hz)')
plt.ylabel('磁芯损耗 (W/m³)')
plt.legend()
plt.title('不同温度下磁芯损耗与频率的关系 (材料1 - 正弦波)')
plt.show()


# Step 4: 构造修正后的斯坦麦茨方程，添加温度影响
def fit_corrected_steinmetz(data):
    frequency = data['频率'].values
    core_loss = data['磁芯损耗'].values
    temperature = data['温度'].values

    # 拟合修正后的斯坦麦茨方程
    popt_corrected, _ = curve_fit(lambda f, k, alpha, beta, n: corrected_steinmetz(f, temperature, k, alpha, beta, n),
                                  frequency, core_loss)

    return popt_corrected


# 拟合修正方程参数
popt_corrected = fit_corrected_steinmetz(sine_wave_data)

# Step 5: 对比斯坦麦茨方程与修正后的方程的预测误差
from sklearn.metrics import mean_squared_error

# 计算原斯坦麦茨方程的预测误差
original_loss = steinmetz(sine_wave_data['频率'].values, *popt_steinmetz)
original_mse = mean_squared_error(sine_wave_data['磁芯损耗'], original_loss)

# 计算修正方程的预测误差
corrected_loss = corrected_steinmetz(sine_wave_data['频率'].values, sine_wave_data['温度'].values, *popt_corrected)
corrected_mse = mean_squared_error(sine_wave_data['磁芯损耗'], corrected_loss)

print(f"斯坦麦茨方程 MSE: {original_mse}")
print(f"修正后的斯坦麦茨方程 MSE: {corrected_mse}")
