import pandas as pd
import matplotlib.pyplot as plt

# 替换为你的文件路径
file_path = '预测错误的样本_附件三.xlsx'

# 读取数据
test_data = pd.read_excel(file_path)

time = [i  for i in range(1024)]  # 生成时间轴（以秒为单位）

plt.figure(figsize=(12, 8))

for idx in range(2):  # 假设绘制前5个样本
    plt.plot(time, test_data.iloc[idx, 5:1029], label=f'样本序号 {test_data.iloc[idx, 0]}')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
plt.title('附件二中多个样本的磁通密度变化曲线')
# plt.xlabel('时间 (s)')
plt.ylabel('磁通密度 (T)')
plt.grid()
plt.legend()
plt.show()