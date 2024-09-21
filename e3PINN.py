import pandas as pd
from pingouin import partial_corr

# 读取数据
file_path = '附件一（训练集）修改.xlsx'
data = pd.concat([
    pd.read_excel(file_path, sheet_name='材料1').assign(材料类型=1),
    pd.read_excel(file_path, sheet_name='材料2').assign(材料类型=2).replace({'励磁波形': {'正弦波': 1, '三角波': 2, '梯形波': 3}}),
    pd.read_excel(file_path, sheet_name='材料3').assign(材料类型=3).replace({'励磁波形': {'正弦波': 1, '三角波': 2, '梯形波': 3}}),
    pd.read_excel(file_path, sheet_name='材料4').assign(材料类型=4).replace({'励磁波形': {'正弦波': 1, '三角波': 2, '梯形波': 3}})
])

# 只保留相关列
data = data[['温度，oC', '频率，Hz', '励磁波形', '磁芯损耗，w/m3', '材料类型']]

# 独立分析偏相关系数
pc_temp = partial_corr(data, x='温度，oC', y='磁芯损耗，w/m3', covar=['励磁波形', '材料类型'])
pc_waveform = partial_corr(data, x='励磁波形', y='磁芯损耗，w/m3', covar=['温度，oC', '材料类型'])
pc_material = partial_corr(data, x='材料类型', y='磁芯损耗，w/m3', covar=['温度，oC', '励磁波形'])

# 协同影响分析（两两组合）
pc_temp_waveform = partial_corr(data, x='温度，oC', y='磁芯损耗，w/m3', covar=['材料类型'])
pc_temp_material = partial_corr(data, x='温度，oC', y='磁芯损耗，w/m3', covar=['励磁波形'])
pc_waveform_material = partial_corr(data, x='励磁波形', y='磁芯损耗，w/m3', covar=['温度，oC'])

# 输出分析结果
print("温度与磁芯损耗的偏相关系数:\n", pc_temp)
print("励磁波形与磁芯损耗的偏相关系数:\n", pc_waveform)
print("材料类型与磁芯损耗的偏相关系数:\n", pc_material)
print("温度和励磁波形协同影响与磁芯损耗的偏相关系数:\n", pc_temp_waveform)
print("温度和材料类型协同影响与磁芯损耗的偏相关系数:\n", pc_temp_material)
print("励磁波形和材料类型协同影响与磁芯损耗的偏相关系数:\n", pc_waveform_material)

# 优化以最小化磁芯损耗
from scipy.optimize import minimize

# 定义目标函数，利用偏相关系数近似模拟磁芯损耗
def objective_function(params):
    temp, waveform, material = params
    loss = (pc_temp['r'][0] * temp +
            pc_waveform['r'][0] * waveform +
            pc_material['r'][0] * material)
    return loss

# 初始猜测
initial_guess = [50, 2, 2]

# 优化
result = minimize(objective_function, initial_guess, bounds=[(20, 90), (1, 3), (1, 4)])
print("最小磁芯损耗的条件:", result.x)
print("最小磁芯损耗值（近似）:", result.fun)
