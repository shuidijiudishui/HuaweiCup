import pandas as pd

# 假设df已经通过pd.read_excel加载
df = pd.read_excel('附件一（训练集）.xlsx')

# 获取第1724行的数据，并从第五个元素开始
row_1_data = df.iloc[1724].tolist()[4:]

# 定义一个函数来判断斜率突变
def is_slope_change(data, threshold):
    m = 0
    for i in range(1, len(data) - 1):
        # 计算相邻两点的斜率
        slope1 = (data[i + 1] - data[i]) / 1
        slope2 = (data[i] - data[i - 1]) / 1
        print(abs(slope1 - slope2))
        # 判断斜率变化是否大于阈值
        if abs(slope1 - slope2) > threshold:
            m += 1
    return m

# 定义一个函数来处理数据
def analyze_wave(data, threshold):
    first_value = data[0]
    second_point = None
    third_point = None
    second_slope_change_count = 0
    third_slope_change_count = 0

    # 寻找第二个点
    for i in range(1, len(data)):
        if abs(data[i] - first_value) <= 0.005 and i - 0 > 10:
            second_point = i
            break

    # 寻找第三个点
    if second_point is not None:
        for i in range(second_point + 1, len(data)):
            if abs(data[i] - first_value) <= 0.005 and i - second_point > 10:
                third_point = i
                break
            else:
                third_point = len(data) -1

    # 计算第二个点和第一个点之间的斜率突变
    if second_point is not None:
        second_slope_change_count = is_slope_change(data[1:second_point], threshold)

    # 计算第三个点和第二个点之间的斜率突变
    if third_point is not None:
        third_slope_change_count = is_slope_change(data[second_point + 1:third_point], threshold)


    print('第二个点', second_point)
    print('第三个点', third_point)
    print('一二之间斜率突变的点有', second_slope_change_count)
    print('二三之间斜率突变的点有', third_slope_change_count)
    # 根据斜率突变的数量判断波形
    if second_slope_change_count == 0:
        print("这是正弦波")
    elif second_slope_change_count == 2:
        print("这是梯形波")
        if third_point is not None and third_slope_change_count == 2:
            print("这是梯形波")
    else:
        print("这是三角波")

# 设置斜率变化的阈值
threshold = 0.01  # 这个值可以根据实际情况调整

# 调用函数
analyze_wave(row_1_data, threshold)