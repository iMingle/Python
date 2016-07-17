
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
    with open('somefile.txt') as f:
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
