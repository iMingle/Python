"""NumPy

"""
import numpy as np


# 数组
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b[1, 1] = 10
print('数组')
print(a.shape)
print(b.shape)
print(a.dtype)
print(b.dtype)
print(b)

# 结构化数组
persontype = np.dtype({
    'names': ['name', 'age', 'chinese', 'math', 'english'],
    'formats': ['S32', 'i', 'i', 'i', 'f']})
peoples = np.array([('ZhangFei', 32, 75, 100, 90), ('GuanYu', 24, 85, 96, 88.5),
                    ('ZhaoYun', 28, 85, 92, 96.5), ('HuangZhong', 29, 65, 85, 100)],
                   dtype=persontype)
ages = peoples[:]['age']
chineses = peoples[:]['chinese']
maths = peoples[:]['math']
englishs = peoples[:]['english']
print('结构化数组')
print(peoples)
print(np.mean(ages))
print(np.mean(chineses))
print(np.mean(maths))
print(np.mean(englishs))

# 算数运算
x1 = np.arange(1, 11, 2)
x2 = np.linspace(1, 9, 5)
print('算数运算')
print(f'x1 = {x1}')
print(f'x2 = {x2}')
print(f'数组加法值 = {np.add(x1, x2)}')
print(np.subtract(x1, x2))
print(np.multiply(x1, x2))
print(np.divide(x1, x2))
print(np.power(x1, x2))
print(np.remainder(x1, x2))

# 统计运算
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('二维数组最大最小值')
print(a)
print(np.amin(a))
print(np.amin(a, 0))
print(np.amin(a, 1))
print(np.amax(a))
print(np.amax(a, 0))
print(np.amax(a, 1))
print(a.sum(axis=1))
print('统计最大值和最小值之差')
print(np.ptp(a))
print(np.ptp(a, 0))
print(np.ptp(a, 1))
print('统计数组的百分位数,第p个百分位数是这样一个值,它使得至少有p%的数据项小于或等于这个值,且至少有(100-p)%的数据项大于或等于这个值')
print(np.percentile(a, 50))
print(np.percentile(a, 50, axis=0))
print(np.percentile(a, 50, axis=1))
print('求中位数')
print(np.median(a))
print(np.median(a, axis=0))
print(np.median(a, axis=1))
print('求平均数')
print(np.mean(a))
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))
print('统计数组中的加权平均值')
a = np.array([1, 2, 3, 4])
wts = np.array([1, 2, 3, 4])
print(a)
print(np.average(a))
print(np.average(a, weights=wts))
print('统计数组中标准差和方差')
print(np.std(a))
print(np.var(a))

# 排序
a = np.array([[4, 3, 2], [2, 4, 1]])
print("排序")
print(a)
print(np.sort(a))
print(np.sort(a, axis=None))
print(np.sort(a, axis=0))
print(np.sort(a, axis=1))
