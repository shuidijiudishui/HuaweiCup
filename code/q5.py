import pandas as pd
import numpy as np
import joblib
from deap import base, creator, tools, algorithms
import random

# 加载训练好的模型
lgb_model = joblib.load('lgb_model.pkl')

# Step 1: 读取数据，并根据 sheet 名添加“材料”列
file_path = '附件一（训练集）.xlsx'
sheets = ['材料1', '材料2', '材料3', '材料4']

# 初始化列表以存储所有数据
all_data = []

# 遍历每个 sheet（代表不同的材料）
for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['材料'] = sheet  # 添加“材料”列，值为 sheet 名
    all_data.append(data)

# 将所有材料的数据合并为一个 DataFrame
all_materials_data = pd.concat(all_data, ignore_index=True)

# Step 2: 数据预处理
# 提取温度、频率、磁芯损耗列和波形列
temperature = all_materials_data.iloc[:, 0].values  # 温度列
frequency = all_materials_data.iloc[:, 1].values    # 频率列
core_loss = all_materials_data.iloc[:, 2].values    # 磁芯损耗列
waveform = all_materials_data.iloc[:, 3].values     # 励磁波形类型
material = all_materials_data['材料'].values        # 提取材料列

# Step 3: 计算磁通密度分布的统计特征
B_columns = all_materials_data.iloc[:, 4:1029].apply(pd.to_numeric, errors='coerce')

# 计算峰值磁通密度 (B_max)
B_max = B_columns.max(axis=1).values

# 计算磁通密度的均值、标准差、峰峰值、偏度、峰度
B_mean = B_columns.mean(axis=1).values
B_std = B_columns.std(axis=1).values
B_peak_to_peak = (B_columns.max(axis=1) - B_columns.min(axis=1)).values
B_skew = B_columns.skew(axis=1).values
B_kurtosis = B_columns.kurtosis(axis=1).values

# Step 4: 对材料类型和励磁波形进行 one-hot 编码
material_encoded = pd.get_dummies(material, prefix='材料')
waveform_encoded = pd.get_dummies(waveform, prefix='波形')
print("Waveform encoded shape:", waveform_encoded.shape)


# 构建特征矩阵，包括原始的温度、频率、B_max，以及提取的磁通密度分布特征
X = np.column_stack((temperature, frequency, B_max, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis))
X = np.hstack((X, material_encoded.values, waveform_encoded.values))

# 获取特征的取值范围，用于约束
temperature_range = (25, 90)
frequency_range = (50000, 500000)
B_max_range = (B_max.min(), B_max.max())
B_mean_range = (B_mean.min(), B_mean.max())
B_std_range = (B_std.min(), B_std.max())
B_peak_to_peak_range = (B_peak_to_peak.min(), B_peak_to_peak.max())
B_skew_range = (B_skew.min(), B_skew.max())
B_kurtosis_range = (B_kurtosis.min(), B_kurtosis.max())
material_code_range = (0, material_encoded.shape[1] - 1)
waveform_code_range = (0, waveform_encoded.shape[1] - 1)

# 定义适应度函数，最小化磁芯损耗
# 定义适应度函数，最小化磁芯损耗
def evaluate(individual):
    if len(individual) != 10:
        raise ValueError(f"Invalid individual length: {len(individual)}")
    print(f"Evaluating individual: {individual}")

    # 解码个体
    temperature = individual[0]
    frequency = individual[1]
    B_max = individual[2]
    B_mean = individual[3]
    B_std = individual[4]
    B_peak_to_peak = individual[5]
    B_skew = individual[6]
    B_kurtosis = individual[7]
    material_code = int(individual[8])
    waveform_code = int(individual[9])

    # 创建 one-hot 编码
    material_one_hot = np.zeros(material_encoded.shape[1])
    waveform_one_hot = np.zeros(waveform_encoded.shape[1])
    material_one_hot[material_code] = 1
    waveform_one_hot[waveform_code] = 1

    # 构建输入特征
    input_features = np.array([temperature, frequency, B_max, B_mean, B_std, B_peak_to_peak, B_skew, B_kurtosis])
    full_input = np.hstack((input_features, material_one_hot, waveform_one_hot)).reshape(1, -1)

    # 预测磁芯损耗
    predicted_loss = lgb_model.predict(full_input)[0]

    # 计算传输磁能
    transmission_energy = frequency * B_max

    # 定义综合目标：最小化磁芯损耗，最大化传输磁能
    fitness = predicted_loss - 0.00001 * transmission_energy

    return (fitness,)

def custom_mutate(individual):
    if len(individual) != 10:
        raise ValueError(f"Invalid individual length during mutation: {len(individual)}")

    # 对每个连续变量进行变异
    mutation_strength = 0.05  # 变异强度，可以根据需要调整
    individual[0] += random.uniform(-mutation_strength, mutation_strength) * (temperature_range[1] - temperature_range[0])
    individual[1] += random.uniform(-mutation_strength, mutation_strength) * (frequency_range[1] - frequency_range[0])
    individual[2] += random.uniform(-mutation_strength, mutation_strength) * (B_max_range[1] - B_max_range[0])
    individual[3] += random.uniform(-mutation_strength, mutation_strength) * (B_mean_range[1] - B_mean_range[0])
    individual[4] += random.uniform(-mutation_strength, mutation_strength) * (B_std_range[1] - B_std_range[0])
    individual[5] += random.uniform(-mutation_strength, mutation_strength) * (B_peak_to_peak_range[1] - B_peak_to_peak_range[0])
    individual[6] += random.uniform(-mutation_strength, mutation_strength) * (B_skew_range[1] - B_skew_range[0])
    individual[7] += random.uniform(-mutation_strength, mutation_strength) * (B_kurtosis_range[1] - B_kurtosis_range[0])

    # 随机选择一个位置并进行变异
    if random.random() < 0.5:
        # 对 material_code 随机重置变异
        individual[8] = random.randint(*material_code_range)
    else:
        # 对 waveform_code 随机重置变异
        individual[9] = random.randint(*waveform_code_range)
    return individual,



# 设置遗传算法的参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 定义个体生成函数
toolbox.register("attr_temperature", random.uniform, *temperature_range)
toolbox.register("attr_frequency", random.uniform, *frequency_range)
toolbox.register("attr_B_max", random.uniform, *B_max_range)
toolbox.register("attr_B_mean", random.uniform, *B_mean_range)
toolbox.register("attr_B_std", random.uniform, *B_std_range)
toolbox.register("attr_B_peak_to_peak", random.uniform, *B_peak_to_peak_range)
toolbox.register("attr_B_skew", random.uniform, *B_skew_range)
toolbox.register("attr_B_kurtosis", random.uniform, *B_kurtosis_range)
toolbox.register("attr_material_code", random.randint, *material_code_range)
toolbox.register("attr_waveform_code", random.randint, *waveform_code_range)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_temperature,
                  toolbox.attr_frequency,
                  toolbox.attr_B_max,
                  toolbox.attr_B_mean,
                  toolbox.attr_B_std,
                  toolbox.attr_B_peak_to_peak,
                  toolbox.attr_B_skew,
                  toolbox.attr_B_kurtosis,
                  toolbox.attr_material_code,
                  toolbox.attr_waveform_code), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传算法操作
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择

# 设置参数
population_size = 50
num_generations = 1000

# 初始化种群
pop = toolbox.population(n=population_size)

# 统计数据
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("mean", np.mean)

# 运行遗传算法
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=stats, verbose=True)

# 获取最优解
best_individual = tools.selBest(pop, k=1)[0]
print('最佳个体：', best_individual)
print('最小磁芯损耗：', evaluate(best_individual)[0])

# 解码最佳个体的变量值
temperature_opt, frequency_opt, B_max_opt, B_mean_opt, B_std_opt, B_peak_to_peak_opt, B_skew_opt, B_kurtosis_opt, material_code_opt, waveform_code_opt = best_individual

print("最优条件：")
print(f"温度: {temperature_opt}")
print(f"频率: {frequency_opt}")
print(f"B_max: {B_max_opt}")
print(f"B_mean: {B_mean_opt}")
print(f"B_std: {B_std_opt}")
print(f"B_peak_to_peak: {B_peak_to_peak_opt}")
print(f"B_skew: {B_skew_opt}")
print(f"B_kurtosis: {B_kurtosis_opt}")
print(f"材料编码: {material_code_opt}")
print(f"波形编码: {waveform_code_opt}")
