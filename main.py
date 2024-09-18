import numpy as np

# 创建一个一维数组
array_1d = np.array([1, 2, 3, 4, 5])

# 创建一个二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 计算数组的总和
sum_1d = np.sum(array_1d)
sum_2d = np.sum(array_2d)

# 计算数组的平均值
mean_1d = np.mean(array_1d)
mean_2d = np.mean(array_2d)

# 计算数组的标准差
std_1d = np.std(array_1d)
std_2d = np.std(array_2d)

# 打印结果
print("一维数组总和:", sum_1d)
print("二维数组总和:", sum_2d)
print("一维数组平均值:", mean_1d)
print("二维数组平均值:", mean_2d)
print("一维数组标准差:", std_1d)
print("二维数组标准差:", std_2d)

