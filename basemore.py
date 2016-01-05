# Copyright (c) 2016, Mingle. All rights reserved.
# Author: Mingle
# Contact: jinminglei@yeah.net

# 方法传送元组
def get_error_details():
	return (2, "second error details")

errnum, errstr = get_error_details()
print(errnum, errstr)

a, *b = [1, 2, 3, 4]
print(b)

a = 5
b = 8
a, b = b, a
print("a =", a, ", b =", b)

# 单语句块,不建议使用
if True: print("True")

# Lambda表达式,用来创建新的函数对象,只能使用表达式
def make_repeater(n):
	return lambda s: s*n

twice = make_repeater(2)
print(twice("work"))
print(twice(5))

points = [{"x": 2, "y": 3}, {"x": 4, "y": 1}]
points.sort(lambda x, y : cmp(x["x"], y["x"]), key = x)
print(points)