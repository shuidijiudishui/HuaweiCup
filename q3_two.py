import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 读取 Excel 文件中的所有表
file_path = '附件一（训练集）.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# 保存所有表的数据
all_data = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df['材料'] = sheet  # 添加材料列
    all_data.append(df)

# 合并所有数据
merged_df = pd.concat(all_data, ignore_index=True)

# 进行双因素方差分析（以温度和材料为例）
anova_df_material_temp = merged_df[['温度，oC', '材料', '磁芯损耗，w/m3']].dropna()

# 使用ols构建线性模型，并包括交互项
model_material_temp = ols('Q("磁芯损耗，w/m3") ~ C(Q("温度，oC")) + C(材料) + C(Q("温度，oC")):C(材料)', data=anova_df_material_temp).fit()

# 进行ANOVA分析
anova_results_material_temp = anova_lm(model_material_temp, typ=2)
print("\n温度与材料的双因素方差分析结果：")
print(anova_results_material_temp)

# 进行其他因素的双因素方差分析，类似地
# 例如材料和频率
anova_df_freq_material = merged_df[['频率，Hz', '材料', '磁芯损耗，w/m3']].dropna()
model_material_freq = ols('Q("磁芯损耗，w/m3") ~ C(Q("频率，Hz")) + C(材料) + C(Q("频率，Hz")):C(材料)', data=anova_df_freq_material).fit()
anova_results_material_freq = anova_lm(model_material_freq, typ=2)
print("\n频率与材料的双因素方差分析结果：")
print(anova_results_material_freq)

# 进行温度和励磁波形的双因素方差分析
anova_df_temp_waveform = merged_df[['温度，oC', '励磁波形', '磁芯损耗，w/m3']].dropna()
model_temp_waveform = ols('Q("磁芯损耗，w/m3") ~ C(Q("温度，oC")) + C(励磁波形) + C(Q("温度，oC")):C(励磁波形)', data=anova_df_temp_waveform).fit()
anova_results_temp_waveform = anova_lm(model_temp_waveform, typ=2)
print("\n温度与励磁波形的双因素方差分析结果：")
print(anova_results_temp_waveform)

# 进行温度和频率的双因素方差分析
anova_df_temp_freq = merged_df[['温度，oC', '频率，Hz', '磁芯损耗，w/m3']].dropna()
model_temp_freq = ols('Q("磁芯损耗，w/m3") ~ C(Q("温度，oC")) + C(Q("频率，Hz")) + C(Q("温度，oC")):C(Q("频率，Hz"))', data=anova_df_temp_freq).fit()
anova_results_temp_freq = anova_lm(model_temp_freq, typ=2)
print("\n温度与频率的双因素方差分析结果：")
print(anova_results_temp_freq)

# 进行材料和励磁波形的双因素方差分析
anova_df_material_waveform = merged_df[['材料', '励磁波形', '磁芯损耗，w/m3']].dropna()
model_material_waveform = ols('Q("磁芯损耗，w/m3") ~ C(材料) + C(励磁波形) + C(材料):C(励磁波形)', data=anova_df_material_waveform).fit()
anova_results_material_waveform = anova_lm(model_material_waveform, typ=2)
print("\n材料与励磁波形的双因素方差分析结果：")
print(anova_results_material_waveform)

# 进行频率和励磁波形的双因素方差分析
anova_df_freq_waveform = merged_df[['频率，Hz', '励磁波形', '磁芯损耗，w/m3']].dropna()
model_freq_waveform = ols('Q("磁芯损耗，w/m3") ~ C(Q("频率，Hz")) + C(励磁波形) + C(Q("频率，Hz")):C(励磁波形)', data=anova_df_freq_waveform).fit()
anova_results_freq_waveform = anova_lm(model_freq_waveform, typ=2)
print("\n频率与励磁波形的双因素方差分析结果：")
print(anova_results_freq_waveform)

