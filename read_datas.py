import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('附件一（训练集）.xlsx')
# print(df)
row_1_data = df.iloc[1724]
row_1_data = list(row_1_data)
row_1_data = row_1_data[4:]
x = [i for i in range(len(row_1_data))]
plt.figure()
plt.plot(x, row_1_data, marker='o')
plt.show()