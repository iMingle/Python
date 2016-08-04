"""Numbers, Dates, and Times.

"""

#1 对数值进行取整
print(round(1.23, 1))
print(round(1.27, 1))
print(round(-1.27, 1))
print(round(1.25361, 3))

a = 1627731
print(round(a, -1))
print(round(a, -2))
print(round(a, -3))

x = 1.23456
print(format(x, '0.2f'))
print(format(x, '.3f'))
print('value is {:.3f}'.format(x))

a = 2.1
b = 4.2
c = a + b
print(c)
c = round(c, 2)
print(c)

#2 执行精确的小数计算
from decimal import Decimal
a = Decimal('4.2')
b = Decimal('2.1')
print(a + b)
print(Decimal('6.3') == (a + b))

from decimal import localcontext
a = Decimal('1.3')
b = Decimal('1.7')
print(a / b)
with localcontext() as ctx:
	ctx.prec = 3
	print(a / b)
with localcontext() as ctx:
	ctx.prec = 50
	print(a / b)

# 大数和小数相加可能出错
nums = [1.23e+18, 1, -1.23e+18]
print(sum(nums))
import math
print(math.fsum(nums))

#3 对数值做格式化输出
# 格式化格式: [<>^]?width[,]?(.digits)?
x = 1234.56789
x = Decimal('1234.56789')
print(format(x, '.2f'))
print(format(-x, '.2f'))
print(format(x, '>10.1f'))
print(format(x, '<10.1f'))
print(format(x, '^10.1f'))
print(format(x, ','))
print(format(x, ',.1f'))
print(format(x, '0,.1f'))
print(format(x, 'e'))
print(format(x, '.2e'))

print('The value is {:0,.2f}'.format(x))

# ,和.转换
swap_separators = {ord('.'): ',', ord(','): '.'}
print(format(x, ',').translate(swap_separators))

print('%0.2f' % x)
print('%10.1f' % x)
print('%-10.1f' % x)

#4 同二进制,八进制和十六进制数打交道
x = 1234
print(bin(x))
print(oct(x))
print(hex(x))

# 不出现0x,0o和0b前缀
print(format(x, 'b'))
print(format(x, 'o'))
print(format(x, 'x'))

x = -1234
print(format(x, 'b'))
print(format(x, 'o'))
print(format(x, 'x'))
# 转为无符号数
print(format(2**32 + x, 'b'))
print(format(2**32 + x, 'o'))
print(format(2**32 + x, 'x'))

print(int('4d2', 16))
print(int('10011010010', 2))

import os
os.chmod('number_date_time.py', 0o755)

#5 从字符串中打包和解包大整数
data = b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
print(len(data))
print(int.from_bytes(data, 'little'))
print(int.from_bytes(data, 'big'))

x = 94522842520747284487117727783387188
print(x.to_bytes(16, 'big'))
print(x.to_bytes(16, 'little'))

# 解包
import struct
hi, lo = struct.unpack('>QQ', data)
print((hi << 64) + lo)

# 确认字节序是大端还是小端
x = 0x01020304
print(x.to_bytes(4, 'big'))
print(x.to_bytes(4, 'little'))

x = 523 ** 23
print(x)
# print(x.to_bytes(16, 'little')) # error,字节太小
print(x.bit_length())
nbytes, rem = divmod(x.bit_length(), 8)
if rem:
	nbytes += 1
print(x.to_bytes(nbytes, 'little'))

#6 复数运算
a = complex(2, 4)
b = 3 - 5j
print(a)
print(b)
print(a.real)
print(a.imag)
print(a.conjugate())
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(abs(a))

import cmath
print(cmath.sin(a))
print(cmath.cos(a))
print(cmath.exp(a))

import numpy as np
a = np.array([2 + 3j, 4 + 5j, 6 - 7j, 8 + 9j])
print(a)
print(a + 2)
print(np.sin(a))

# print(math.sqrt(-1)) # error
print(cmath.sqrt(-1))

#7 处理无穷大和NaN
a = float('inf') # 正无穷大
b = float('-inf') # 负无穷大
c = float('nan') # not a number
print(math.isinf(a))
print(math.isinf(b))
print(math.isnan(c))

# 某些操作产生NaN,NaN会通过所有的操作进行传播,且不会引发任何异常
print(a + 45)
print(a / a)
print(a + b)
print(c + 23)
print(c / 2)
print(c * 2)
print(math.sqrt(c))

d = float('nan')
print(c == d) # False

#8 分数的计算
from fractions import Fraction
a = Fraction(5, 4)
print(a)
b = Fraction(7, 16)
print(a + b)
print(a * b)
c = a * b
print(c.numerator)
print(c.denominator)
print(float(c))

# limit the denominator of a value
print(c.limit_denominator(8))

x = 3.75
y = Fraction(*x.as_integer_ratio())
print(y)

#9 处理大型数组的计算
# 对于任何涉及数组的计算密集型任务,请使用NumPy库.
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
print(x * 2)
# print(x + 10) # error
print(x + y)

ax = np.array([1, 2, 3, 4])
ay = np.array([5, 6, 7, 8])
print(ax * 2)
print(ax + 10)
print(ax + ay)

def f(x):
	return 3*x**2 - 2*x + 7
print(f(ax))

print(np.sqrt(ax))
print(np.cos(ax))

# 创建大型数组
grid = np.zeros(shape=(1000, 10000), dtype=float)
print(grid)
print(grid + 10)
print(np.sin(grid + 10))

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[1])
# select column 1
print(a[:,1])
# select subregion and change it
print(a[1:3, 1:3])
a[1:3, 1:3] += 10
print(a)
print(a + [100, 101, 102, 103])
print(np.where(a < 10, a, 10))

#10 矩阵和线性代数的计算
m = np.matrix([[1, -2, 3], [0, 4, 5], [7, 8, -9]])
print(m)
# return transpose
print(m.T)
# return inverse
print(m.I)

v = np.matrix([[2], [3], [4]])
print(v)
print(m * v)

import numpy.linalg
# determinant
print(numpy.linalg.det(m))
# eigenvalues
print(numpy.linalg.eigvals(m))
# solve for x in mx = v
x = numpy.linalg.solve(m, v)
print(x)
print(m * x)
print(v)

#11 随机选择
import random
values = [1, 2, 3, 4, 5, 6]
print(random.choice(values))
print(random.choice(values))
print(random.choice(values))
print(random.choice(values))
print(random.choice(values))
# 取样出N个元素
print(random.sample(values, 2))
print(random.sample(values, 2))
print(random.sample(values, 3))
print(random.sample(values, 3))

random.shuffle(values)
print(values)
random.shuffle(values)
print(values)

print(random.randint(0, 10))
print(random.randint(0, 10))
print(random.randint(0, 10))
print(random.randint(0, 10))
print(random.randint(0, 10))

print(random.random())
print(random.random())
print(random.random())
print(random.random())

# 得到由N个随机比特位所表示的整数
print(random.getrandbits(200))

# random模块采用马特塞特旋转算法,是确定算法,可以通过random.seed()函数
# 来修改初始的种子值
random.seed() # seed based on system time or os.urandom()
random.seed(12345)
random.seed(b'bytedata')

#12 时间换算
from datetime import timedelta # 时间间隔
a = timedelta(days=2, hours=6)
b = timedelta(hours=4.5)
c = a + b
print(c.days)
print(c.seconds)
print(c.seconds / 3600)
print(c.total_seconds() / 3600)

from datetime import datetime
a = datetime(2016, 7, 1)
print(a + timedelta(days=10))
b = datetime(2016, 8, 8)
d = b - a
print(d.days)
now = datetime.today()
print(now)
print(now + timedelta(minutes=10))

# 处理闰年
a = datetime(2016, 3, 1)
b = datetime(2016, 2, 28)
print(a - b)
print((a - b).days)
c = datetime(2017, 3, 1)
d = datetime(2017, 2, 28)
print((c - d).days)

# 处理复杂日期问题
a = datetime(2016, 9, 23)
# print(a + timedelta(months=1)) error
from dateutil.relativedelta import relativedelta
print(a + relativedelta(months=+1))
print(a + relativedelta(months=+4))
# time between two dates
b = datetime(2016, 12, 21)
d = b - a
print(d)
d = relativedelta(b, a)
print(d)
print(d.months)
print(d.days)

#13 计算上周五的日期
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_previous_byday(dayname, start_date=None):
    if start_date is None:
        start_date = datetime.today()
    day_num = start_date.weekday()
    day_num_target = weekdays.index(dayname)
    days_ago = (7 + day_num - day_num_target) % 7
    if days_ago == 0:
        days_ago = 7
    target_date = start_date - timedelta(days=days_ago)
    return target_date
print(datetime.today())
print(get_previous_byday('Monday'))
print(get_previous_byday('Tuesday'))
print(get_previous_byday('Wednesday'))
print(get_previous_byday('Friday'))
print(get_previous_byday('Sunday'))

from dateutil.rrule import *
d = datetime.now()
print(d)
# next friday
print(d + relativedelta(weekday=FR))
# last friday
print(d + relativedelta(weekday=FR(-1)))

#14 计算当前月份的日期范围
from datetime import date
import calendar

def get_month_range(start_date=None):
    if start_date is None:
        start_date = date.today().replace(day=1)
    _, days_in_month = calendar.monthrange(start_date.year, start_date.month)
    end_date = start_date + timedelta(days=days_in_month)
    return (start_date, end_date)
a_day = timedelta(days=1)
first_day, last_day = get_month_range()
while first_day < last_day:
    print(first_day)
    first_day += a_day

def date_range(start, stop, step):
    while start < stop:
        yield start
        start += step
for d in date_range(datetime(2016, 9, 1), datetime(2016, 10, 1), timedelta(hours=6)):
    print(d)

#15 字符串转换为日期
text = '2016-07-28'
y = datetime.strptime(text, '%Y-%m-%d') # strptime性能比较差
z = datetime.now()
diff = z - y
print(diff)
# 日期转换为字符串
nice_z = datetime.strftime(z, '%A %B %d, %Y')
print(nice_z)

def parse_ymd(s):
    year, month, day = s.split('-')
    return datetime(int(year), int(month), int(day))
print(parse_ymd('2016-07-30'))

#16 结合时区的日期操作
# 对几乎所有涉及到时区的问题,你都应该使用pytz模块.这个包提供了Olson时
# 区数据库,它是时区信息的事实上的标准,在很多语言和操作系统里面都可以找到.
from pytz import timezone
d = datetime(2016, 7, 27, 19, 30, 30)
print(d)
# Localize the date for Chicago
central = timezone('US/Central')
loc_d = central.localize(d)
print(loc_d)
# Convert to Bangalore time
bang_d = loc_d.astimezone(timezone('Asia/Kolkata'))
print(bang_d)
d = datetime(2013, 3, 10, 1, 45)
loc_d = central.localize(d)
print(loc_d)
later = loc_d + timedelta(minutes=30)
print(later)
later = central.normalize(loc_d + timedelta(minutes=30))
print(later)

# 为了不让你被这些东东弄的晕头转向,处理本地化日期的通常的策略先将所有日期
# 转换为UTC时间,并用它来执行所有的中间存储和操作.
print(loc_d)
import pytz
utc_d = loc_d.astimezone(pytz.utc)
print(utc_d)
later_utc = utc_d + timedelta(minutes=30)
print(later_utc.astimezone(central))

# 获取时区的名称
print(pytz.country_timezones['CN'])

# python 3.2 加入timezone
from datetime import timezone
d = datetime(2016, 7, 28, tzinfo=timezone.utc)
print(d)
