import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

# 读取 Excel 文件
df = pd.read_excel('附件一（训练集）.xlsx')

# 1. 计算相同温度下磁芯损耗的平均值和标准差
temp_stats = df.groupby('温度，oC')['磁芯损耗，w/m3'].agg(['mean', 'std']).reset_index()
temp_stats.columns = ['温度', '平均磁芯损耗', '磁芯损耗标准差']

# 2. 计算相同频率下磁芯损耗的平均值和标准差
freq_stats = df.groupby('频率，Hz')['磁芯损耗，w/m3'].agg(['mean', 'std']).reset_index()
freq_stats.columns = ['频率', '平均磁芯损耗', '磁芯损耗标准差']

# 3. 计算相同励磁波形下磁芯损耗的平均值和标准差
waveform_stats = df.groupby('励磁波形')['磁芯损耗，w/m3'].agg(['mean', 'std']).reset_index()
waveform_stats.columns = ['励磁波形', '平均磁芯损耗', '磁芯损耗标准差']

# 读取 Excel 文件

# 获取所有表格的名称
file_path = '附件一（训练集）.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# 用于保存各个表的统计结果
results = []

# 遍历每个表格
for sheet in sheet_names:
    # 读取表格
    df = pd.read_excel(file_path, sheet_name=sheet)

    # 假设表中有一列为 '磁芯损耗'，计算平均值和标准差
    avg_loss = df['磁芯损耗，w/m3'].mean()
    std_loss = df['磁芯损耗，w/m3'].std()

    # 保存结果
    results.append({
        '材料': sheet,  # 表名作为材料名称
        '平均磁芯损耗': avg_loss,
        '磁芯损耗标准差': std_loss
    })

print("材料统计：")
# 将结果转换为 DataFrame 以便输出
result_df = pd.DataFrame(results)
print(result_df)

# 输出结果
print("温度统计：")
print(temp_stats)
print("\n频率统计：")
print(freq_stats)
print("\n励磁波形统计：")
print(waveform_stats)

sns.set(style="whitegrid")

# 1. 画出温度 vs 磁芯损耗平均值与标准差的图
plt.figure(figsize=(10, 6))
plt.errorbar(temp_stats['温度'], temp_stats['平均磁芯损耗'], yerr=temp_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('Temperature vs Core loss (mean and standard deviation)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Core loss')
plt.xticks(sorted(temp_stats['温度']))  # 确保横轴升序
plt.grid(True)
plt.show()

# 2. 画出频率 vs 磁芯损耗平均值与标准差的图
plt.figure(figsize=(10, 6))
plt.errorbar(freq_stats['频率'], freq_stats['平均磁芯损耗'], yerr=freq_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('frequency vs Core loss (mean and standard deviation)')
plt.xlabel('frequency (Hz)')
plt.ylabel('Core loss')
plt.xticks(sorted(freq_stats['频率']))  # 确保横轴升序
plt.grid(True)
plt.show()

# 3. 画出励磁波形 vs 磁芯损耗平均值与标准差的图
plt.figure(figsize=(10, 6))
plt.errorbar(waveform_stats['励磁波形'], waveform_stats['平均磁芯损耗'], yerr=waveform_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('Excitation waveform vs Core loss (mean and standard deviation)')
plt.xlabel('Excitation waveform')
plt.ylabel('Core loss')
plt.xticks(sorted(waveform_stats['励磁波形']))  # 确保横轴升序
plt.grid(True)
plt.show()

# 4. 画出材料 vs 磁芯损耗平均值与标准差的图
# 构造示例数据
data = {
    '材料': ['material1', 'material2', 'material3', 'material4'],
    '平均磁芯损耗': [179886.944216, 234317.144307, 264453.073238, 109469.249064],
    '磁芯损耗标准差': [339525.652644, 409095.679288, 465459.562564, 213889.831057]
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)
# 设置图表风格
sns.set(style="whitegrid")
# 绘制柱状图，带有标准差的误差条
plt.figure(figsize=(10, 6))
plt.bar(df['材料'], df['平均磁芯损耗'], yerr=df['磁芯损耗标准差'], capsize=5, color='skyblue')
plt.title('Core losses for different materials (mean and standard deviation)')
plt.xlabel('material')
plt.ylabel('Core loss')
plt.grid(axis='y')  # 在y轴上添加网格线
plt.show()

# 分析两两之间对Core loss的影响
# 1. 创建一个新的包含所有变量的 DataFrame，假设有材料、温度、频率和励磁波形列
df_combined = pd.concat([temp_stats, freq_stats, waveform_stats, result_df], axis=1)

# 2. 进行多元线性回归，带有交互项
# 回归公式为磁芯损耗与材料、温度、频率、励磁波形及它们的交互项
formula = 'Q("平均磁芯损耗") ~ C(材料) * C(温度) + C(频率) * C(励磁波形)'
model = ols(formula, data=df_combined).fit()

# 3. 输出回归结果
print(model.summary())

# 4. 绘制交互作用图
import seaborn as sns

# 交互作用分析 - 材料与温度的交互作用图
plt.figure(figsize=(10, 6))
sns.lmplot(x='温度', y='平均磁芯损耗', hue='材料', data=temp_stats, ci=None)
plt.title('材料与温度对磁芯损耗的交互作用')
plt.grid(True)
plt.show()

# 交互作用分析 - 频率与励磁波形的交互作用图
plt.figure(figsize=(10, 6))
sns.lmplot(x='频率', y='平均磁芯损耗', hue='励磁波形', data=freq_stats, ci=None)
plt.title('频率与励磁波形对磁芯损耗的交互作用')
plt.grid(True)
plt.show()