import pandas as pd

df = pd.read_excel('附件一（训练集）.xlsx')
filtered_df = df[df['励磁波形'] == '正弦波']
first_part = filtered_df.iloc[:, :4]
second_part = filtered_df.iloc[:, 4:]
max_values = second_part.max(axis=1)
third_part = pd.DataFrame(max_values, columns=['峰值'])
all_data = pd.concat([first_part, third_part], axis=1)
