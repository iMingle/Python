"""Functions.

"""

#1 编写可接受任意数量参数的函数
def avg(first, *rest):
    return (first + sum(rest)) / (1 + len(rest))
print(avg(1, 2))
print(avg(1, 2, 3, 4))

import html

def make_element(name, value, **attrs):
    keyvals = [" %s='%s'" % item for item in attrs.items()]
    attr_str = ''.join(keyvals)
    element = '<{name}{attrs}>{value}</{name}>'.format(name=name, attrs=attr_str, value=html.escape(value))
    return element
e = make_element('item', 'Albatross', size='large', quantity=6)
print(e)
e = make_element('p', '<spam>')
print(e)

def anyargs(*args, **kwargs):
    print(args) # a tuple
    print(kwargs) # a dict

#2 编写只接受关键字参数的函数
# A simple keyword-only argument
def recv(maxsize, *, block=True):
    print(maxsize, block)

recv(8192, block=False) # Works
try:
    recv(8192, False) # Fails
except TypeError as e:
    print(e)

# Adding keyword-only args to *args functions
def minimum(*values, clip=None):
    m = min(values)
    if clip is not None:
        m = clip if clip > m else m
    return m

print(minimum(1, 5, 2, -5, 10))
print(minimum(1, 5, 2, -5, 10, clip=0))

#3 将元数据信息附加到函数参数上
def add(x:int, y:int) -> int:
    return x + y
print(add.__annotations__)

#4 从函数中返回多个值
def fun():
    return 1, 2, 3 # 元组
a, b, c = fun()
print(a, b, c)

#5 定义带有默认参数的函数
def spam(a, b=42):
    print(a, b)
spam(1)
spam(1, 2)

# 如果默认值是可变容器的话,比如列表,集合,或者字典,那么应该把None作为默认值
def spam(a, b=None):
    if b is None:
        b = []
    print(a, b)
spam(1)
spam(1, 2)

x = 42
# 默认值在函数定义的时候就确定好了
def spam(a, b=x):
    print(a, b)
spam(1)
x = 23 # 不起作用
spam(1)

#6 定义匿名或内联函数
add = lambda x, y: x + y
print(add(2, 3))
print(add('hello', 'world'))

names = ['David Beazley', 'Brian Jones', 'Raymond Hettinger', 'Ned Batchelder']
print(sorted(names, key=lambda name: name.split()[-1].lower()))

#7 在匿名函数中绑定变量的值
x = 10
a = lambda y: x + y
x = 20
b = lambda y: x + y
print(a(10)) # 30,运行时绑定
print(b(10))

x = 10
a = lambda y, x=x: x + y
x = 20
b = lambda y, x=x: x + y
print(a(10))
print(b(10))

func = [lambda x: x+n for n in range(5)]
for f in func:
    print(f(0)) # 4
func = [lambda x, n=n: x+n for n in range(5)]
for f in func:
    print(f(0)) # 0, 1, 2, 3, 4

#8 让带有N个参数的可调用对象以较少的参数形式调用
def spam(a, b, c, d):
    print(a, b, c, d)
from functools import partial
s1 = partial(spam, 1)
s1(2, 3, 4)
s1(4, 5, 6)
s2 = partial(spam, d=42)
s2(1, 2, 3)
s2(4, 5, 5)
s3 = partial(spam, 1, 2, d=42)
s3(3)
s3(4)
s3(5)

points = [(1, 2), (3, 4), (5, 6), (7, 8)]

import math
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)

pt = (4, 3)
points.sort(key=partial(distance, pt))
print(points)

def output_result(result, log=None):
    if log is None:
        log.debug('Got: %r', result)

def add(x, y):
    return x + y

if '__main__' == __name__:
    import logging
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('test')

    p = Pool()
    p.apply_async(add, (3, 4), callback=partial(output_result, log=log))
    p.close()
    p.join()

from socketserver import StreamRequestHandler, TCPServer

class EchoHandler(StreamRequestHandler):
    """echo服务程序"""
    def __init__(self, *args, ack, **kwargs):
        self.ack = ack
        super(EchoHandler, self).__init__()

    def handle(self):
        for line in self.rfile:
            self.wfile.write(b'GOT:' + line)
serv = TCPServer(('', 15000), partial(EchoHandler, ack=b'RECEIVED:'))
# serv.serve_forever()

#9 用函数替代只有单个方法的类
# 在许多情况下,只有单个方法的类可以通过闭包将其转换为函数
from urllib.request import urlopen

class UrlTemplate:
    def __init__(self, template):
        super(UrlTemplate, self).__init__()
        self.template = template

    def open(self, **kwargs):
        return urlopen(self.template.format_map(kwargs))

yahoo = UrlTemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo.open(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

# 闭包
def urltemplate(template):
    def opener(**kwargs):
        return urlopen(template.format_map(kwargs))
    return opener
yahoo = urltemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

#10 在回调函数中携带额外的状态
def apply_async(func, args, *, callback):
    # compute the result
    result = func(*args)
    # invoke the callback with the result
    callback(result)

def print_result(result):
    print('GOT:', result)

def add(x, y):
    return x + y

apply_async(add, (2, 3), callback=print_result)
apply_async(add, ('hello', 'world'), callback=print_result)

class ResultHandler:
    def __init__(self):
        self.sequence = 0

    def handler(self, result):
        self.sequence += 1
        print('[{}] Got: {}'.format(self.sequence, result))
r = ResultHandler()
apply_async(add, (2, 3), callback=r.handler)

# 闭包
def make_handler():
    sequence = 0
    def handler(result):
        nonlocal sequence
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
    return handler
handler = make_handler()
apply_async(add, (2, 3), callback=handler)
apply_async(add, ('hello', 'world'), callback=handler)

# 协程coroutine
def make_handler():
    sequence = 0
    while True:
        result = yield
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
handler = make_handler()
next(handler)
apply_async(add, (2, 3), callback=handler.send)
apply_async(add, ('hello', 'world'), callback=handler.send)

# partial
class SequenceNo:
    def __init__(self):
        self.sequence = 0

def handler(result, seq):
    seq.sequence += 1
    print('[{}] Got: {}'.format(seq.sequence, result))
seq = SequenceNo()
apply_async(add, (2, 3), callback=partial(handler, seq=seq))
apply_async(add, ('hello', 'world'), callback=partial(handler, seq=seq))
apply_async(add, ('hello', 'world'), callback=lambda r: handler(r, seq))

#11 内联回调函数
from queue import Queue
from functools import wraps

class Async:
    def __init__(self, func, args):
        super(Async, self).__init__()
        self.func = func
        self.args = args

def inlined_async(func):
    @wraps(func)
    def wrapper(*args):
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            result = result_queue.get()
            try:
                a = f.send(result)
                apply_async(a.func, a.args, callback=result_queue.put)
            except StopIteration:
                break
    return wrapper

def add(x, y):
    return x + y

@inlined_async
def test():
    r = yield Async(add, (3, 4))
    print(r)
    r = yield Async(add, ('hello', 'world'))
    print(r)
    for n in range(10):
        r = yield Async(add, (n, n))
        print(r)
    print('goodbye')
if '__main__' == __name__:
    # Simple test
    print('# --- Simple test')
    test()

    print('# --- Multiprocessing test')
    import multiprocessing
    pool = multiprocessing.Pool()
    apply_async = pool.apply_async
    # test()

#12 访问定义在闭包内的变量
def sample():
    n = 0
    def func():
        print('n=', n)

    def get_n():
        return n

    def set_n(value):
        nonlocal n
        n = value

    func.get_n = get_n
    func.set_n = set_n
    return func

f = sample()
f()
f.set_n(10)
f()
print(f.get_n())

import sys
# 闭包模拟类实例比较快
class ClosureIntance:
    """闭包"""
    def __init__(self, locals=None):
        if locals is None:
            locals = sys._getframe(1).f_locals
        # update instance dictionary with callables
        self.__dict__.update((key, value) for key, value in locals.items() if callable(value))

    def __len__(self):
        return self.__dict__['__len__']()

def Stack():
    items = []

    def push(item):
        items.append(item)

    def pop():
        return items.pop()

    def __len__():
        return len(items)

    return ClosureIntance()
s = Stack()
print(s)
s.push(10)
s.push(20)
s.push('hello')
print(len(s))
print(s.pop())
print(s.pop())
print(s.pop())

class Stack2:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def __len__(self):
        return len(self.items)
from timeit import timeit
try:
    s = Stack()
    print('Stack', timeit('s.push(1);s.pop()', 'from __main__ import s'))
    s = Stack2()
    print('Stack2', timeit('s.push(1);s.pop()', 'from __main__ import s'))
except Exception:
    print('timeit error')
