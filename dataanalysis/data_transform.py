"""数据变换

"""
from sklearn import preprocessing
import numpy as np


# 初始化数据，每一行表示一个样本，每一列表示一个特征
x = np.array([[0., -3., 1.],
              [3., 1., 2.],
              [0., 1., -1.]])
# min-max: 新数值 =（原数值 - 极小值）/（极大值 - 极小值）
# 将数据进行[0,1]规范化
min_max_scaler = preprocessing.MinMaxScaler()
minmax_x = min_max_scaler.fit_transform(x)
print(minmax_x)

# z-score: 新数值 =（原数值 - 均值）/ 标准差
# 将数据进行Z-Score规范化
scaled_x = preprocessing.scale(x)
print(scaled_x)

# 小数定标规范化: 通过移动小数点的位置来进行规范化,小数点移动多少位取决于属性A的取值中的最大绝对值
j = np.ceil(np.log10(np.max(abs(x))))
scaled_x = x / (10 ** j)
print(scaled_x)
