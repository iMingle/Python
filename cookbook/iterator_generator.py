"""Iterators and Generators.

"""

#1 手动遍历迭代器
def manual_iter():
    with open('data/somefile.txt') as f:
        try:
            while True:
                line = next(f)
                print(line, end='')
        except StopIteration:
            pass
manual_iter()

def manual_iter_new():
    with open('data/somefile.txt') as f:
            while True:
                line = next(f, None)
                if line is None:
                    break
                print(line, end='')
manual_iter_new()

items = [1, 2, 3]
it = iter(items)
print(next(it))
print(next(it))
print(next(it))

#2 委托迭代
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value) # !r表示调用repr()

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)
if '__main__' == __name__:
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)

    for ch in root:
        print(ch)

#3 用生成器创建新的迭代模式
def frange(start, stop, increment):
    """产生某个范围的浮点数"""
    x = start
    while x < stop:
        yield x
        x += increment
for n in frange(0, 4, 0.5):
    print(n)

def countdown(n):
    print('Starting to count down', n)
    while n > 0:
        yield n
        n -= 1
    print('Done!')
c = countdown(3)
print(c)
print(next(c))
print(next(c))
print(next(c))
# print(next(c)) # StopIteration

#4 实现迭代协议
class Node:
    """树结构"""
    def __init__(self, value):
        super(Node, self).__init__()
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        """深度优先"""
        yield self
        for c in self:
            yield from c.depth_first()

if '__main__' == __name__:
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))
    for ch in root.depth_first():
        print(ch)

class Node:
    """树结构"""
    def __init__(self, value):
        super(Node, self).__init__()
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        return DepthFirstIterator(self)

class DepthFirstIterator(object):
    """深度优先迭代器"""
    def __init__(self, start_node):
        super(DepthFirstIterator, self).__init__()
        self._node = start_node
        self._children_iter = None
        self._child_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        """费解"""
        # return myself if just started, create an iterator for children
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        # if processing a child, return its next item
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        # advance to the next child and start its iteration
        else:
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)
if '__main__' == __name__:
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))
    for ch in root.depth_first():
        print(ch)

#5 反向迭代
a = [1, 2, 3, 4]
for x in reversed(a):
    print(x)

f = open('data/somefile.txt')
for line in reversed(list(f)):
    print(line, end='')

class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

    def __reversed__(self):
        n = 1
        while n < self.start:
            yield n
            n += 1

#6 定义带有额外状态的生成器函数
from collections import deque
class linehistory:
    def __init__(self, lines, histlen=3):
        super(linehistory, self).__init__()
        self.lines = lines
        self.history = deque(maxlen=histlen)

    def __iter__(self):
        for lineno, line in enumerate(self.lines, 1):
            self.history.append((lineno, line))
            yield line

    def clear(self):
        self.history.clear()
with open('data/somefile.txt') as f:
    lines = linehistory(f)
    print(lines)
    for line in lines:
        if 'python' in line:
            for lineno, hline in lines.history:
                print('{}:{}'.format(lineno, hline), end='')

#7 对迭代器做切片操作
def count(n):
    while True:
        yield n
        n += 1
c = count(0)
# print(c[10:20]) # error
import itertools
for x in itertools.islice(c, 10, 20):
    print(x)

#8 跳过可迭代对象中的前一部分元素
from itertools import dropwhile, islice
with open('data/somefile.txt') as f:
    for line in dropwhile(lambda line: line.startswith('='), f):
        print(line, end='')

items = ['a', 'b', 'c', 1, 4, 10, 15]
for x in islice(items, 3, None):
    print(x)

#9 迭代所有可能的组合或排列
items = ['a', 'b', 'c']
from itertools import permutations
for p in permutations(items):
    print(p)
for p in permutations(items, 2):
    print(p)

from itertools import combinations
for c in combinations(items, 3):
    print(c)
for c in combinations(items, 2):
    print(c)
for c in combinations(items, 1):
    print(c)

from itertools import combinations_with_replacement
for c in combinations_with_replacement(items, 3):
    print(c)

#10 以索引-值对的形式迭代序列
my_list = ['a', 'b', 'c']
for idx, value in enumerate(my_list):
    print(idx, value)
for idx, value in enumerate(my_list, 1):
    print(idx, value)

def parse_data(filename):
    with open(filename, 'rt') as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split()
            try:
                count = int(fields[1])
            except ValueError as e:
                print('Line {}: Parse error: {}'.format(lineno, e))
# parse_data('data/somefile.txt')

from collections import defaultdict
word_summary = defaultdict(list)
with open('data/somefile.txt') as f:
    lines = f.readlines()
for idx, line in enumerate(lines):
    words = [w.strip().lower() for w in line.split()]
    for word in words:
        word_summary[word].append(idx)
print(word_summary)

#11 同时迭代多个序列
xpts = [1, 5, 4, 2, 10, 7]
ypts = [101, 78, 37, 15, 62, 99]
for x, y in zip(xpts, ypts):
    print(x, y)
a = [1, 2, 3]
b = ['a', 'b', 'c', 'd']
for i in zip(a, b):
    print(i)

from itertools import zip_longest
for i in zip_longest(a, b):
    print(i)
for i in zip_longest(a, b, fillvalue=0):
    print(i)

headers = ['name', 'shares', 'price']
values = ['ACME', 100, 490.1]
s = dict(zip(headers, values))
print(s)
for name, value in zip(headers, values):
    print(name, '=', value)

#12 在不同的迭代器中进行迭代
from itertools import chain
a = [1, 2, 3, 4]
b = ['x', 'y', 'z']
for x in chain(a, b):
    print(x)

active_items = set()
inactive_items = set()
for item in chain(active_items, inactive_items):
    print(item)

#13 创建处理数据的管道
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    """find all filenames in a directory tree that match a shell wildcard pattern."""
    for dirpath, dirnames, filenames in os.walk(top):
        for name in fnmatch.filter(filenames, filepat):
            yield os.path.join(dirpath, name)

def gen_opener(filenames):
    """open a sequence of filenames one at a time producing a file object.
    the file is closed immediately when proceeding to the next iteration.
    """
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        yield f
        f.close()

def gen_concatenate(iterators):
    """chain a sequence of iterators together into a single sequence."""
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    """look for a regex pattern in a sequence of lines"""
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line
if __name__ == '__main__':
    lognames = gen_find('access-log*', 'log')
    files = gen_opener(lognames)
    lines = gen_concatenate(files)
    pylines = gen_grep('(?i)python', lines) # ignore case
    for line in pylines:
        print(line)
    # 统计总字节量
    lognames = gen_find('access-log*', 'log')
    files = gen_opener(lognames)
    lines = gen_concatenate(files)
    pylines = gen_grep('(?i)python', lines)
    bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
    bytesTuple = (int(x) for x in bytecolumn if x != '-')
    print('Total', sum(bytesTuple))

#14 扁平化处理嵌套型的序列
from collections import Iterable

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

items = [1, 2, [3, 4, [5, 6], 7], 8]
for x in flatten(items):
    print(x)
items = ['Dave', 'Paula', ['Thomas', 'Lewis']]
for x in flatten(items):
    print(x)

#15 合并多个有序序列,再对整个有序序列进行迭代
import heapq
a = [1, 4, 7, 10]
b = [2, 5, 6, 11]
# heapq.merge()要求所有的输入序列是有序的
for c in heapq.merge(a, b):
    print(c)

#16 用迭代器取代while循环
CHUNKSIZE = 8192

def reader(s):
    while True:
        data = s.recv(CHUNKSIZE)
        if b'' == data:
            break
        process_data(data)

def reader_iter(s):
    for chunk in iter(lambda: s.recv(CHUNKSIZE), b''):
        process_data(data)

import sys
f = open('data/somefile.txt')
for chunk in iter(lambda: f.read(10), ''):
    n = sys.stdout.write(chunk)
