import pandas as pd
import matplotlib.pyplot as plt

# 替换为你的文件路径
file_path = '附件一（训练集）.xlsx'

# 读取四个数据表
material1_data = pd.read_excel(file_path, sheet_name='材料1')
material2_data = pd.read_excel(file_path, sheet_name='材料2')
material3_data = pd.read_excel(file_path, sheet_name='材料3')
material4_data = pd.read_excel(file_path, sheet_name='材料4')

time = [i * 1e-6 for i in range(1024)]  # 生成时间轴（以秒为单位）

plt.figure(figsize=(12, 8))
for material_data, label in zip([material1_data, material2_data, material3_data, material4_data],
                                 ['材料1', '材料2', '材料3', '材料4']):
    # 假设选取第一行作为示例
    plt.plot(time, material_data.iloc[2900, 4:1028], label=label)  # 从第5列开始提取磁通密度

plt.title('不同材料的磁通密度随时间变化曲线')
plt.xlabel('时间 (s)')
plt.ylabel('磁通密度 (T)')
plt.grid()
plt.legend()
plt.show()