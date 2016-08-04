"""Data Structures and Algorithms.

"""

#1 将序列分解为单独的变量
p = (4, 5)
x, y = p
print(x)
print(y)

data = ['ACME', 50, 91.1, (2016, 7, 17)]
name, shares, price, date = data
print(name)
print(date)

name, shares, price, (year, month, day) = data
print(name)
print(year)

# 可迭代的
s = 'hello'
a, b, c, d, e = s
print(a)
print(d)

_, shares, price, _ = data
print(shares)
print(price)

#2 在任意长度的可迭代对象中分解元素
def avg(l):
    return sum(l, 0.0) / len(l)

def drop_first_last(grades):
    first, *middle, last = grades
    return avg(middle)
print(drop_first_last([3, 4, 5, 6, 7]))

record = ('Mingle', 'mingle@yeah.net', '0372-1234567', '010-1234567')
name, email, *phone_numbers = record
print(phone_numbers)

records = [
    ('foo', 1, 2),
    ('bar', 'hello'),
    ('foo', 3, 4)
]

def do_foo(x, y):
    print('foo', x, y)

def do_bar(s):
    print('bar', s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)

line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'
uname, *fields, homedir, sh = line.split(':')
print(uname)
print(homedir)
print(sh)

record = ['ACME', 50, 91.1, (2016, 7, 17)]
name, *_, (*_, day) = record
print(name)
print(day)

items = [1, 10, 7, 4, 5, 9]
head, *tail = items
print(head)
print(tail)

def sum(items):
    head, *tail = items
    return head + sum(tail) if tail else head

print(sum(items))

#3 保存最后N个元素
from collections import deque

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)

if __name__ == '__main__':
    with open('data/somefile.txt') as f:
        for line, prevlines in search(f, 'python', 5):
            for pline in prevlines:
                print(pline, end = '')
            print(line, end = '')
            print('-' * 20)

q = deque(maxlen=3)
q.append(1)
q.append(2)
q.append(3)
print(q)
q.append(4)
print(q)
q.append(5)
print(q)

q = deque()
q.append(1)
q.append(2)
q.append(3)
print(q)
q.appendleft(4)
print(q)
print(q.pop())
print(q)
print(q.popleft())

#4 找到最大或最小的N个元素
import heapq

nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums))
print(heapq.nsmallest(3, nums))
heap = list(nums)
heapq.heapify(heap)
print(heap)
print(heapq.heappop(heap))
print(heapq.heappop(heap))
print(heapq.heappop(heap))

portfolio = [
   {'name': 'IBM', 'shares': 100, 'price': 91.1},
   {'name': 'AAPL', 'shares': 50, 'price': 543.22},
   {'name': 'FB', 'shares': 200, 'price': 21.09},
   {'name': 'HPQ', 'shares': 35, 'price': 31.75},
   {'name': 'YHOO', 'shares': 45, 'price': 16.35},
   {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(cheap)
print(expensive)

#5 实现优先级队列
class PriorityQueue(object):
    """优先级队列"""
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Item:
    """how to use PriorityQueue"""
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return 'Item({!s})'.format(self.name)

q = PriorityQueue()
q.push(Item('foo'), 1)
q.push(Item('bar'), 5)
q.push(Item('spam'), 4)
q.push(Item('grok'), 1)
print(q.pop())
print(q.pop())
print(q.pop())
print(q.pop())

#6 在字典中将键映射到多个值上
listMap = {
    'a': [1, 2, 3],
    'b': [4, 5]
}

setMap = {
    'a': {1, 2, 3},
    'b': {4, 5}
}

from collections import defaultdict

d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)
d['b'].append(4)
print(d)

d = defaultdict(set)
d['a'].add(1)
d['a'].add(2)
d['b'].add(2)
print(d)

#7 让字典保持有序
from collections import OrderedDict

d = OrderedDict()
d['foo'] = 1
d['bar'] = 2
d['spam'] = 3
d['grok'] = 4
for key in d:
    print(key, d[key])

import json

print(json.dumps(d))

#8 与字典有关的计算问题
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
min_price = min(zip(prices.values(), prices.keys()))
print(min_price)
max_price = max(zip(prices.values(), prices.keys()))
print(max_price)

print(min(prices))
print(min(prices.values()))
print(min(prices, key=lambda k: prices[k]))
min_value = prices[min(prices, key=lambda k: prices[k])]
print(min_value)

#9 在两个字典中寻找相同点
a = {
    'x': 1,
    'y': 2,
    'z': 3
}

b = {
    'w': 10,
    'x': 11,
    'y': 2
}

print(a.keys() & b.keys())
print(a.keys() - b.keys())
print(a.items() & b.items())

c = {key:a[key] for key in a.keys() - {'z', 'w'}}
print(c)

#10 从序列中移除重复项且保持元素间顺序不变

def dedupe(items):
    """序列中的值是可哈希的"""
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

a = [1, 5, 2, 1, 9, 1, 5, 10]
print(list(dedupe(a)))

def dedupe(items, key=None):
    """序列中的值不用必须的可哈希的"""
    seen = set()
    for item in items:
        value = item if key is None else key(item)
        if value not in seen:
            yield item
            seen.add(value)

a = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 2}, {'x': 2, 'y': 4}]
print(list(dedupe(a, key=lambda d: (d['x'], d['y']))))
print(list(dedupe(a, key=lambda d: d['x'])))

#11 对切片命名
record = '.....100.....513.25.....'
cost = int(record[5:7]) * float(record[13:18])
print(cost)
SHARES = slice(5, 7)
PRICE= slice(13, 18)
cost = int(record[SHARES]) * float(record[PRICE])
print(cost)

items = [0, 1, 2, 3, 4, 5, 6]
a = slice(2, 4)
print(items[2:4])
print(items[a])
items[a] = [10, 11]
print(items)
del items[a]
print(items)
print(a.start)
print(a.stop)
print(a.step)

s = 'HelloWorld'
print(a.indices(len(s)))
print(*a.indices(len(s)))
for i in range(*a.indices(len(s))):
    print(s[i])

#12 找出序列中出现次数最多的元素
words = [
   'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
   'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
   'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
   'my', 'eyes', "you're", 'under'
]

from collections import Counter

word_counts = Counter(words)
top_three = word_counts.most_common(3)
print(top_three)
print(word_counts['not'])
print(word_counts['eyes'])

morewords = ['why','are','you','not','looking','in','my','eyes']
word_counts.update(morewords)
top_three = word_counts.most_common(3)
print(top_three)

a = Counter(words)
b = Counter(morewords)
print(a)
print(b)
c = a + b
print(c)
d = a - b
print(d)

#13 通过公共键对字典列表排序
rows = [
    {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
    {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
    {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
    {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}
]

from operator import itemgetter

rows_by_fname = sorted(rows, key=itemgetter('fname'))
rows_by_uid = sorted(rows, key=itemgetter('uid'))
print(rows_by_fname)
print(rows_by_uid)
rows_by_lfname = sorted(rows, key=itemgetter('lname', 'fname'))
print(rows_by_lfname)

#14 对不支持原生比较操作的对象排序
class User:
    def __init__(self, user_id):
        super(User, self).__init__()
        self.user_id = user_id

    def __repr__(self):
        return 'User({})'.format(self.user_id)

users = [User(23), User(3), User(99)]
print(users)
print(sorted(users, key=lambda u: u.user_id))

from operator import attrgetter
print(sorted(users, key=attrgetter('user_id'))) # 优先使用

#15 根据字段将记录分组
rows = [
    {'address': '5412 N CLARK', 'date': '07/01/2012'},
    {'address': '5148 N CLARK', 'date': '07/04/2012'},
    {'address': '5800 E 58TH', 'date': '07/02/2012'},
    {'address': '2122 N CLARK', 'date': '07/03/2012'},
    {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
    {'address': '1060 W ADDISON', 'date': '07/02/2012'},
    {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
    {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
]

from itertools import groupby

rows.sort(key=itemgetter('date'))
print(rows)
for date, items in groupby(rows, key=itemgetter('date')):
    print(date)
    for i in items:
        print('    ', i)

#16 筛选序列中的元素
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
pos = [n for n in mylist if n > 0]
print(pos)
neg = [n for n in mylist if n < 0]
print(neg)
# Negative values clipped to 0
neg_clip = [n if n > 0 else 0 for n in mylist]
print(neg_clip)
# Positive values clipped to 0
pos_clip = [n if n < 0 else 0 for n in mylist]
print(pos_clip)

values = ['1', '2', '-3', '-', '4', 'N/A', '5']

def is_int(value):
    try:
        x = int(value)
        return True
    except ValueError:
        return False

ivals = list(filter(is_int, values))
print(ivals)

import math

print([math.sqrt(n) for n in mylist if n > 0])

addresses = [
    '5412 N CLARK',
    '5148 N CLARK', 
    '5800 E 58TH',
    '2122 N CLARK',
    '5645 N RAVENSWOOD',
    '1060 W ADDISON',
    '4801 N BROADWAY',
    '1039 W GRANVILLE',
]

counts = [ 0, 3, 10, 4, 1, 7, 6, 1]

from itertools import compress

more5 = [n > 5 for n in counts]
print(more5)
print(list(compress(addresses, more5)))

#17 从字典中提取子集
from pprint import pprint

prices = {
   'ACME': 45.23,
   'AAPL': 612.78,
   'IBM': 205.55,
   'HPQ': 37.20,
   'FB': 10.75
}

p1 = {key:value for key, value in prices.items() if value > 200} # 优先使用
print(p1)
p1 = dict((key, value) for key, value in prices.items() if value > 200)
print(p1)
tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
p2 = {key:value for key, value in prices.items() if key in tech_names}
print(p2)

#18 将名称映射到序列的元素中
from collections import namedtuple

# namedtuple不可变
subscriber = namedtuple('subscriber', ['addr', 'joined'])
sub = subscriber('mingle@yeah.net', '2016-07-21')
print(sub)
print(sub.addr)
print(sub.joined)
print(len(sub))
addr, joined = sub
print(addr)
print(joined)

def compute_cost(records):
    """使用普通元组"""
    total = 0.0
    for rec in records:
        total += rec[1] * rec[2]
    return total

stock = namedtuple('stock', ['name', 'shares', 'price'])
def compute_cost_namedtuple(records):
    total = 0.0
    for rec in records:
        s = stock(*rec)
        total += s.shares * s.price
    return total

s = stock('ACME', 100, 123.45)
print(s)
# s.shares = 75 # error
s = s._replace(shares=75)
print(s)

stock = namedtuple('stock', ['name', 'shares', 'price', 'date', 'time'])
stock_prototype = stock('', 0, 0.0, None, None)
def dict_to_stock(s):
    return stock_prototype._replace(**s)

a = {'name': 'ACME', 'shares': 100, 'price': 123.45}
print(dict_to_stock(a))
b = {'name': 'ACME', 'shares': 100, 'price': 123.45, 'date': '2016-07-21'}
print(dict_to_stock(b))

#19 同时对数据做转换和换算
nums = [1, 2, 3, 4, 5]
s = sum(x * x for x in nums)
print(s)

import os
files = os.listdir('.')
if any(name.endswith('.py') for name in files):
    print('There be python!')
else:
    print('Sorry, no python.')

s = ('stock', 50, 123.45)
print(','.join(str(x) for x in s))

portfolio = [
   {'name': 'GOOG', 'shares': 100},
   {'name': 'YHOO', 'shares': 50},
   {'name': 'AOL', 'shares': 200},
   {'name': 'SCOX', 'shares': 35}
]
min_shares = min(s['shares'] for s in portfolio)
print(min_shares)

#20 将多个映射合并为单个映射
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}

from collections import ChainMap

c = ChainMap(a, b)
print(c)
print(c['x'])
print(c['y'])
print(c['z'])
print(len(c))
print(list(c.keys()))
print(list(c.values()))
c['z'] = 10
c['w'] = 40
del c['x']
print(a)
# del c['y'] # error

# 处理不同作用域问题
values = ChainMap()
values['x'] = 1
values = values.new_child()
values['x'] = 2
values = values.new_child()
values['x'] = 3
print(values)
print(values['x'])
values = values.parents
print(values['x'])
values = values.parents
print(values['x'])
print(values)

# 字典合并
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
merged = dict(b)
merged.update(a)
print(merged)
a['x'] = 13
print(merged['x']) # 1

a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
merged = ChainMap(a, b)
print(merged['x'])
a['x'] = 13
print(merged['x']) # 13
