import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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

# 读取 Excel 文件中的所有表
file_path = '附件一（训练集）.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# 保存各个表的统计结果
results = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)

    # 假设表中有一列为 '磁芯损耗'，计算平均值和标准差
    avg_loss = df['磁芯损耗，w/m3'].mean()
    std_loss = df['磁芯损耗，w/m3'].std()

    # 保存结果
    results.append({
        '材料': sheet,
        '平均磁芯损耗': avg_loss,
        '磁芯损耗标准差': std_loss
    })

# 将结果转换为 DataFrame
result_df = pd.DataFrame(results)
print("材料统计：")
print(result_df)
print("温度统计：")
print(temp_stats)
print("\n频率统计：")
print(freq_stats)
print("\n励磁波形统计：")
print(waveform_stats)
# # ================= 双因素方差分析 ====================
#
# # 假设我们要分析温度和频率对磁芯损耗的影响
# # 我们可以提取这些数据并进行双因素方差分析
# anova_df1 = df[['温度，oC', '频率，Hz', '磁芯损耗，w/m3']].dropna()
#
# # 使用ols构建线性模型，并包括交互项
# model = ols('Q("磁芯损耗，w/m3") ~ C(Q("温度，oC")) + C(Q("频率，Hz")) + C(Q("温度，oC")):C(Q("频率，Hz"))', data=anova_df1).fit()
#
# # 进行ANOVA分析
# anova_results = anova_lm(model, typ=2)
# print("\n双因素方差分析结果：")
# print(anova_results)
#
# # 假设我们要分析温度和励磁波形对磁芯损耗的影响
# anova_df2 = df[['温度，oC', '励磁波形', '磁芯损耗，w/m3']].dropna()
#
# # 使用ols构建线性模型，并包括交互项
# model2 = ols('Q("磁芯损耗，w/m3") ~ C(Q("温度，oC")) + C(Q("励磁波形")) + C(Q("温度，oC")):C(Q("励磁波形"))', data=anova_df2).fit()
#
# # 进行ANOVA分析
# anova_results2 = anova_lm(model2, typ=2)
# print("\n温度与励磁波形的双因素方差分析结果：")
# print(anova_results2)
#
# # 假设我们要分析频率和励磁波形对磁芯损耗的影响
# anova_df3 = df[['频率，Hz', '励磁波形', '磁芯损耗，w/m3']].dropna()
#
# # 使用ols构建线性模型，并包括交互项
# model3 = ols('Q("磁芯损耗，w/m3") ~ C(Q("频率，Hz")) + C(Q("励磁波形")) + C(Q("频率，Hz")):C(Q("励磁波形"))', data=anova_df3).fit()
#
# # 进行ANOVA分析
# anova_results3 = anova_lm(model3, typ=2)
# print("\n频率与励磁波形的双因素方差分析结果：")
# print(anova_results3)


# ================ 可视化分析 ===================

sns.set(style="whitegrid")

# 1. 温度 vs 磁芯损耗平均值与标准差
plt.figure(figsize=(10, 6))
plt.errorbar(temp_stats['温度'], temp_stats['平均磁芯损耗'], yerr=temp_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('Temperature vs Core loss (mean and standard deviation)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Core loss')
plt.xticks(sorted(temp_stats['温度']))  # 确保横轴升序
plt.grid(True)
plt.show()

# 2. 频率 vs 磁芯损耗平均值与标准差
plt.figure(figsize=(10, 6))
plt.errorbar(freq_stats['频率'], freq_stats['平均磁芯损耗'], yerr=freq_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('Frequency vs Core loss (mean and standard deviation)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Core loss')
plt.xticks(sorted(freq_stats['频率']))  # 确保横轴升序
plt.grid(True)
plt.show()

# 3. 励磁波形 vs 磁芯损耗平均值与标准差
plt.figure(figsize=(10, 6))
plt.errorbar(waveform_stats['励磁波形'], waveform_stats['平均磁芯损耗'], yerr=waveform_stats['磁芯损耗标准差'], fmt='-o', capsize=5)
plt.title('Excitation waveform vs Core loss (mean and standard deviation)')
plt.xlabel('Excitation waveform')
plt.ylabel('Core loss')
plt.grid(True)
plt.show()

# 4. 材料 vs 磁芯损耗平均值与标准差
plt.figure(figsize=(10, 6))
plt.bar(result_df['材料'], result_df['平均磁芯损耗'], yerr=result_df['磁芯损耗标准差'], capsize=5, color='skyblue')
plt.title('Core losses for different materials (mean and standard deviation)')
plt.xlabel('Material')
plt.ylabel('Core loss')
plt.grid(axis='y')
plt.show()
