"""Python base

"""

# 方法传送元组
def get_error_details():
    return (2, 'second error details')

errnum, errstr = get_error_details()
print(errnum, errstr)

a, *b = [1, 2, 3, 4]
print(b)

a = 5
b = 8
a, b = b, a
print('a =', a, ', b =', b)

# 单语句块,不建议使用
if True: print('True')

# Lambda表达式,用来创建新的函数对象,只能使用表达式
def make_repeater(n):
    return lambda s: s*n

twice = make_repeater(2)
print(twice('work'))
print(twice(5))

points = [{'x': 2, 'y': 3}, {'x': 4, 'y': 1}, {'x': 0, 'y': 1}, {'x': 1, 'y': 1}]
points.sort(key = lambda point : point['x']);
print(points)

# 列表综合
listone = [2, 3, 4]
listtwo = [2*i for i in listone if i > 2]
print(listtwo)

def powersum(power, *args):
    """Return the sum of each argument raised to specified power."""
    total = 0
    for i in args:
        total += pow(i, power)
    return total

print(powersum(2, 3, 4))
print(powersum(2, 10))

# exec语句用来执行储存在字符串或文件中的Python语句
exec("print('hello')")
# eval函数用来执行存储在字符串中的Python表达式
print(eval('2*3'))

# assert
list = ['item']
assert len(list) >= 1
list.pop()
# assert len(list) >= 1

# repr函数用来取得对象的规范字符串表示,注意,在大多数时候有eval(repr(object)) == object.
i = []
i.append('item')
print(i)
print(repr(i))
print(eval(repr(i)))
print(eval(repr(i)) == i)