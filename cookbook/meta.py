"""Metaprogramming.

"""

#1 给函数添加一个包装
import time
from functools import wraps

def timethis(func):
    """decorator that reports the execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper

# 装饰器就是一个函数,它可以接受一个函数作为输入并返回一个新的函数作为输出
@timethis
def countdown(n:int):
    """count down"""
    while n > 0:
        n -= 1

countdown(100000)
countdown(1000000)

#2 编写装饰器时如何保存函数的元数据
# 每当定义一个装饰器时,应该总是记得为底层的包装函数添加functools库的@wraps装饰器
print(countdown.__name__)
print(countdown.__doc__)
print(countdown.__annotations__)
countdown.__wrapped__(1000000) # 直接访问被包装的函数

from inspect import signature
print(signature(countdown))

#3 对包装器进行解包装
@timethis
def add(x, y):
    return x + y

orig_add = add.__wrapped__
print(orig_add(3, 4))

def decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('decorator 1')
        return func(*args, **kwargs)
    return wrapper

def decorator2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('decorator 2')
        return func(*args, **kwargs)
    return wrapper

@decorator1
@decorator2
def add(x, y):
    return x + y

print(add(2, 4))
print(add.__wrapped__(2, 4))

#4 定义一个可接受参数的装饰器
import logging

def logged(level, name=None, message=None):
    """
    Add logging to a function.  level is the logging
    level, name is the logger name, and message is the
    log message.  If name and message aren't specified,
    they default to the function's module and name.
    """
    def decorate(func):
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)
        return wrapper
    return decorate

@logged(logging.DEBUG)
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('spam!')

if '__main__' == __name__:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print(add(3, 4))
    spam()

#5 定义一个属性可由用户修改的装饰器
from functools import partial

def attach_wrapper(obj, func=None):
    """utility decorator to attach a function as an attribute of obj"""
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def logged(level, name=None, message=None):
    def decorate(func):
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)

        # attach setter functions
        @attach_wrapper(wrapper)
        def set_level(newlevel):
            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper)
        def set_message(newmsg):
            nonlocal logmsg
            logmsg = newmsg

        return wrapper
    return decorate

@logged(logging.DEBUG)
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('spam!')

if '__main__' == __name__:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print(add(3, 4))
    spam()
    add.set_message('add called')
    print(add(3, 4))
    add.set_level(logging.WARNING)
    print(add(3, 4))

#6 定义一个能接受可选参数的装饰器
def logged(func=None, *, level=logging.DEBUG, name=None, message=None):
    if func is None:
        return partial(logged, level=level, name=name, message=message)
    logname = name if name else func.__module__
    log = logging.getLogger(logname)
    logmsg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        log.log(level, logmsg)
        return func(*args, **kwargs)
    return wrapper

@logged
def add(x, y):
    return x + y

@logged(level=logging.CRITICAL, name='example')
def spam():
    print('spam!')

if '__main__' == __name__:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print(add(3, 4))
    spam()

#9 利用包装器对函数参数强制执行类型检查
def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        # if in optional mode, disable type checking
        if not __debug__:
            return func
        # map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate

@typeassert(int, z=int)
def spam(x, y, z=42):
    print(x, y, z)

spam(1, 2, 3)
spam(1, 'hello', 3)
# spam(1, 'hello', 'world') # TypeError: argument z must be <class 'int'>
sig = signature(spam);
print(sig)
print(sig.parameters)
print(sig.parameters['z'].name)
print(sig.parameters['z'].default)
print(sig.parameters['z'].kind)
bound_types = sig.bind_partial(int, z=int) # 部分绑定
print(bound_types)
print(bound_types.arguments)
bound_values = sig.bind(1, 2, 3) # 全部绑定
print(bound_values)
print(bound_values.arguments)

@typeassert(int, list)
def bar(x, items=None):
    if items is None:
        items = []
    items.append(x)
    return items

print(bar(2))
# print(bar(2, 3)) # TypeError: argument items must be <class 'list'>

#8 在类中定义装饰器
class A:
    # decorator as an instance method
    def decorator1(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('decorator 1')
            return func(*args, **kwargs)
        return wrapper

    # decorator as a class method
    @classmethod
    def decorator2(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('decorator 2')
            return func(*args, **kwargs)
        return wrapper

# As an instance method
a = A()

@a.decorator1
def spam():
    pass

# As a class method
@A.decorator2
def grok():
    pass

spam()
grok()

#9 把装饰器定义成类
# 要把装饰器定义成类实例,需要确保在类中实现__call__()和__get__()方法
import types

class Profiled:
    def __init__(self, func):
        wraps(func)(self)
        self.ncalls = 0

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)

@Profiled
def add(x, y):
    return x + y

class Spam:
    @Profiled
    def bar(self, x):
        print(self, x)

if '__main__' == __name__:
    print(add(2, 3))
    print(add(4, 5))
    print('ncalls:', add.ncalls)

    s = Spam()
    s.bar(1)
    s.bar(2)
    s.bar(3)
    print('ncalls:', Spam.bar.ncalls)

# Reformulation using closures and function attributes
def profiled(func):
    ncalls = 0
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal ncalls
        ncalls += 1
        return func(*args, **kwargs)
    wrapper.ncalls = lambda: ncalls
    return wrapper

@profiled
def add(x, y):
    return x + y

class Spam:
    @profiled
    def bar(self, x):
        print(self, x)

if '__main__' == __name__:
    print(add(2,3))
    print(add(4,5))
    print('ncalls:', add.ncalls())

    s = Spam()
    s.bar(1)
    s.bar(2)
    s.bar(3)
    print('ncalls:', Spam.bar.ncalls())

#10 把装饰器作用到类和静态方法上
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(end - start)
        return result
    return wrapper

# class illustrating application of the decorator to different kinds of methods
class Spam:
    @timethis
    def instance_method(self, n):
        print(self, n)
        while n > 0:
            n -= 1

    @classmethod
    @timethis
    def class_method(cls, n):
        print(cls, n)
        while n > 0:
            n -= 1

    @staticmethod
    @timethis
    def static_method(n):
        print(n)
        while n > 0:
            n -= 1

if '__main__' == __name__:
    s = Spam()
    s.instance_method(1000000)
    Spam.class_method(1000000)
    Spam.static_method(1000000)

#11 编写装饰器为被包装的函数添加参数
import inspect

def optional_debug(func):
    if 'debug' in inspect.getargspec(func).args:
        raise TypeError('debug argument already defined')

    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print('calling', func.__name__)
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    params.append(inspect.Parameter('debug',
        inspect.Parameter.KEYWORD_ONLY,
        default=False))
    wrapper.__signature__ = sig.replace(parameters=params)

    return wrapper

@optional_debug
def spam(a, b, c):
    print(a, b, c)

spam(1, 2, 3)
spam(1, 2, 3, debug=True)
print(inspect.signature(spam))

#12 利用装饰器给类定义打补丁
def log_getattribute(cls):
    # get the original implementation
    orig_getattribute = cls.__getattribute__

    # make a new definition
    def new_getattribute(self, name):
        print('getting:', name)
        return orig_getattribute(self, name)

    # attach to the class and return
    cls.__getattribute__ = new_getattribute
    return cls

@log_getattribute
class A:
    def __init__(self, x):
        super(A, self).__init__()
        self.x = x

    def spam(self):
        pass

a = A(10)
print(a.x)
a.spam()

#13 利用元类来控制实例的创建
class NoInstances(type):
    """不让创建实例"""
    def __call__(self, *args, **kwargs):
        raise TypeError("can't instantiate directly")

class Spam(metaclass=NoInstances):
    @staticmethod
    def grok(x):
        print('Spam.grok')

if '__main__' == __name__:
    try:
        s = Spam()
    except TypeError as e:
        print(e)

    Spam.grok(42)

class Singleton(type):
    """单例"""
    def __init__(self, *args, **kwargs):
        self._instance = None
        super(Singleton, self).__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = super().__call__(*args, **kwargs)
        return self._instance

class Spam(metaclass=Singleton):
    def __init__(self):
        print('creating spam')

if '__main__' == __name__:
    a = Spam()
    b = Spam()
    print(a is b)

import weakref

class Cached(type):
    """缓存"""
    def __init__(self, *args, **kwargs):
        super(Cached, self).__init__(*args, **kwargs)
        self._cache = weakref.WeakValueDictionary()
    
    def __call__(self, *args):
        if args in self._cache:
            return self._cache[args]
        else:
            obj = super().__call__(*args)
            self._cache[args] = obj
            return obj

class Spam(metaclass=Cached):
    def __init__(self, name):
        print('creating spam({!r})'.format(name))
        self.name = name

if '__main__' == __name__:
    a = Spam('Guido')
    b = Spam('Diana')
    c = Spam('Diana')
    print(a is b)
    print(b is c)

#14 获取类属性的定义顺序
from collections import OrderedDict

class Typed:
    """a set of descriptors for various types"""
    _expected_type = type(None)

    def __init__(self, name=None):
        super(Typed, self).__init__()
        self.name = name
    
    def __set__(self, instance, value):
        if not isinstance(value, self._expected_type):
            raise TypeError('expected ' + str(self._expected_type))
        instance.__dict__[self.name] = value

class Integer(Typed):
    _expected_type = int

class Float(Typed):
    _expected_type = float

class String(Typed):
    _expected_type = str

class OrderedMeta(type):
    """metaclass that uses an OrderedDict for class body"""
    def __new__(cls, clsname, bases, clsdict):
        d = dict(clsdict)
        order = []
        for name, value in clsdict.items():
            if isinstance(value, Typed):
                value._name = name
                order.append(name)
        d['_order'] = order
        return type.__new__(cls, clsname, bases, d)

    @classmethod
    def __prepare__(cls, clsname, bases):
        return OrderedDict()

class Structure(metaclass=OrderedMeta):
    def as_csv(self):
        return ','.join(str(getattr(self, name)) for name in self._order)

class Stock(Structure):
    name = String()
    shares = Integer()
    price = Float()

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

if '__main__' == __name__:
    s = Stock('GOOG', 100, 490.1)
    print(s.name)
    print(s.as_csv())
    try:
        s = Stock('GOOG', 'a lot', 490.1)
    except TypeError as e:
        print(e)

class NoDupOrderedDict(OrderedDict):
    """无重复的字典"""
    def __init__(self, clsname):
        super(NoDupOrderedDict, self).__init__()
        self.clsname = clsname
    
    def __setitem__(self, name, value):
        if name in self:
            raise TypeError('{} already defined in {}'.format(name, self.clsname))
        super().__setitem__(name, value)

class OrderedMeta(type):
    """metaclass that uses an OrderedDict for class body"""
    def __new__(cls, clsname, bases, clsdict):
        d = dict(clsdict)
        order = []
        d['_order'] = [name for name in clsdict if name[0] != '_']
        return type.__new__(cls, clsname, bases, d)

    @classmethod
    def __prepare__(cls, clsname, bases):
        return NoDupOrderedDict(clsname)

try:
    class A(metaclass=OrderedMeta):
        def spam(self):
            pass

        def spam(self):
            pass
except TypeError as e:
    print(e)

#15 定义一个能接受可选参数的元类
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxsize=None):
        pass

    @abstractmethod
    def write(self, data):
        pass

class MyMeta(type):
    # optional
    @classmethod
    def __prepare__(cls, name, bases, *, debug=False, synchronize=False):
        # custom processing
        return super().__prepare__(name, bases)

    # required
    def __new__(cls, name, bases, ns, *, debug=False, synchronize=False):
        # custom processing
        return super().__new__(cls, name, bases, ns)
        
    def __init__(self, name, bases, ns, *, debug=False, synchronize=False):
        # custom processing
        super().__init__(name, bases, ns)

class A(metaclass=MyMeta, debug=True, synchronize=True):
    pass

class B(metaclass=MyMeta):
    pass

class C(metaclass=MyMeta, synchronize=True):
    pass

#16 在*args和**kwargs上强制规定一种参数签名
from inspect import Signature, Parameter

params = [
    Parameter('x', Parameter.POSITIONAL_OR_KEYWORD),
    Parameter('y', Parameter.POSITIONAL_OR_KEYWORD, default=42),
    Parameter('z', Parameter.KEYWORD_ONLY, default=None)
]
sig = Signature(params)
print(sig)

def func(*args, **kwargs):
    bound_values = sig.bind(*args, **kwargs)
    for name, value in bound_values.arguments.items():
        print(name, value)

if '__main__' == __name__:
    func(1, 2, z=3)
    func(1)
    func(1, z=3)
    func(y=2, x=1)
    try:
        func(1, 2, 3, 4)
        func(y=2)
    except TypeError as e:
        print(e)

def make_sig(*names):
    parms = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
             for name in names]
    return Signature(parms)

class Structure:
    __signature__ = make_sig()
    def __init__(self, *args, **kwargs):
        bound_values = self.__signature__.bind(*args, **kwargs)
        for name, value in bound_values.arguments.items():
            setattr(self, name, value)

class Stock(Structure):
    __signature__ = make_sig('name', 'shares', 'price')

class Point(Structure):
    __signature__ = make_sig('x', 'y')

if '__main__' == __name__:
    s1 = Stock('ACME', 100, 490.1)
    print(s1.name, s1.shares, s1.price)

    s2 = Stock(shares=100, name='ACME', price=490.1)
    print(s2.name, s2.shares, s2.price)

    # not enough args
    try:
        s3 = Stock('ACME', 100)
    except TypeError as e:
        print(e)

    # too many args
    try:
        s4 = Stock('ACME', 100, 490.1, '12/21/2012')
    except TypeError as e:
        print(e)

    # replicated args
    try:
        s5 = Stock('ACME', 100, name='ACME', price=490.1)
    except TypeError as e:
        print(e)

class StructureMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        clsdict['__signature__'] = make_sig(*clsdict.get('_fields', []))
        return super().__new__(cls, clsname, bases, clsdict)

class Structure(metaclass=StructureMeta):
    _fields = []

    def __init__(self, *args, **kwargs):
        bound_values = self.__signature__.bind(*args, **kwargs)
        for name, value in bound_values.arguments.items():
            setattr(self, name, value)

if '__main__' == __name__:
    s1 = Stock('ACME', 100, 490.1)
    print(s1.name, s1.shares, s1.price)

    s2 = Stock(shares=100, name='ACME', price=490.1)
    print(s2.name, s2.shares, s2.price)

    # not enough args
    try:
        s3 = Stock('ACME', 100)
    except TypeError as e:
        print(e)

    # too many args
    try:
        s4 = Stock('ACME', 100, 490.1, '12/21/2012')
    except TypeError as e:
        print(e)

    # replicated args
    try:
        s5 = Stock('ACME', 100, name='ACME', price=490.1)
    except TypeError as e:
        print(e)

#17 在类中强制规定编码约定
class NoMixedCaseMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        for name in clsdict:
            if name.lower() != name:
                raise TypeError('Bad attribute name: ' + name)
        return super().__new__(cls, clsname, bases, clsdict)

class Root(metaclass=NoMixedCaseMeta):
    pass

class A(Root):
    def foo_bar(self): # Ok
        pass

print('**** about to generate a TypeError')
try:
    class B(Root):
        def fooBar(self): # TypeError
            pass
except TypeError as e:
    print(e)

class MatchSignaturesMeta(type):
    """检查子类中是否有重新定义的方法"""
    def __init__(self, clsname, bases, clsdict):
        super().__init__(clsname, bases, clsdict)
        sup = super(self, self)
        for name, value in clsdict.items():
            if name.startswith('_') or not callable(value):
                continue
            # get the previous definition (if any) and compare the signatures
            prev_dfn = getattr(sup, name, None)
            if prev_dfn:
                prev_sig = signature(prev_dfn)
                val_sig = signature(value)
                if prev_sig != val_sig:
                    logging.warning('signature mismatch in %s. %s != %s',
                                value.__qualname__, str(prev_sig), str(val_sig))

class Root(metaclass=MatchSignaturesMeta):
    pass

class A(Root):
    def foo(self, x, y):
        pass

    def spam(self, x, *, z):
        pass

# class with redefined methods, but slightly different signatures
class B(A):
    def foo(self, a, b):
        pass

    def spam(self,x,z):
        pass

#18 通过编程的方式来定义类
def __init__(self, name, shares, price):
    self.name = name
    self.shares = shares
    self.price = price

def cost(self):
    return self.shares * self.price

cls_dict = {
    '__init__': __init__,
    'cost': cost
}

Stock = types.new_class('Stock', (), {'metaclass': ABCMeta}, lambda ns: ns.update(cls_dict))
Stock.__module__ = __name__
s = Stock('ACME', 50, 91.1)
print(s)
print(s.cost())
print(type(Stock))

import operator
import sys

def named_tuple(classname, fieldnames):
    # populate a dictionary of field property accessors
    cls_dict = { name: property(operator.itemgetter(n))
                 for n, name in enumerate(fieldnames) }
    print(cls_dict)
    # make a __new__ function and add to the class dict
    def __new__(cls, *args):
        if len(args) != len(fieldnames):
            raise TypeError('expected {} arguments'.format(len(fieldnames)))
        return tuple.__new__(cls, (args))

    cls_dict['__new__'] = __new__

    # make the class
    cls = types.new_class(classname, (tuple,), {}, 
                           lambda ns: ns.update(cls_dict))
    cls.__module__ = sys._getframe(1).f_globals['__name__']
    return cls

if '__main__' == __name__:
    Point = named_tuple('Point', ['x', 'y'])
    print(Point)
    p = Point(4, 5)
    print(len(p))
    print(p.x, p[0])
    print(p.y, p[1])
    try:
        p.x = 2
    except AttributeError as e:
        print(e)
    print('%s %s' % p)

#19 在定义的时候初始化类成员
class StructTupleMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for n, name in enumerate(cls._fields):
            setattr(cls, name, property(operator.itemgetter(n)))

class StructTuple(tuple, metaclass=StructTupleMeta):
    _fields = []

    def __new__(cls, *args):
        if len(args) != len(cls._fields):
            raise ValueError('{} arguments required'.format(len(cls._fields)))
        return super().__new__(cls, args)

class Stock(StructTuple):
    _fields = ['name', 'shares', 'price']

class Point(StructTuple):
    _fields = ['x', 'y']

if '__main__' == __name__:
    s = Stock('ACME', 50, 91.1)
    print(s)
    print(s[0])
    print(s.name)
    print(s.shares * s.price)
    try:
        s.shares = 23
    except AttributeError as e:
        print(e)

#20 通过函数注解来实现方法重载
class MultiMethod:
    """represents a single multimethod."""
    def __init__(self, name):
        self._methods = {}
        self.__name__ = name

    def register(self, meth):
        """register a new method as a multimethod"""
        sig = inspect.signature(meth)

        # build a type-signature from the method's annotations
        types = []
        for name, parm in sig.parameters.items():
            if name == 'self':
                continue
            if parm.annotation is inspect.Parameter.empty:
                raise TypeError('Argument {} must be annotated with a type'.format(name))
            if not isinstance(parm.annotation, type):
                raise TypeError('Argument {} annotation must be a type'.format(name))
            if parm.default is not inspect.Parameter.empty:
                self._methods[tuple(types)] = meth
            types.append(parm.annotation)

        self._methods[tuple(types)] = meth

    def __call__(self, *args):
        """call a method based on type signature of the arguments"""
        types = tuple(type(arg) for arg in args[1:])
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            raise TypeError('No matching method for types {}'.format(types))
        
    def __get__(self, instance, cls):
        """d method needed to make calls work in a class"""
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

class MultiDict(dict):
    """special dictionary to build multimethods in a metaclass"""
    def __setitem__(self, key, value):
        if key in self:
            # if key already exists, it must be a multimethod or callable
            current_value = self[key]
            if isinstance(current_value, MultiMethod):
                current_value.register(value)
            else:
                mvalue = MultiMethod(key)
                mvalue.register(current_value)
                mvalue.register(value)
                super().__setitem__(key, mvalue)
        else:
            super().__setitem__(key, value)

class MultipleMeta(type):
    """metaclass that allows multiple dispatch of methods"""
    def __new__(cls, clsname, bases, clsdict):
        return type.__new__(cls, clsname, bases, dict(clsdict))

    @classmethod
    def __prepare__(cls, clsname, bases):
        return MultiDict()

# some example classes that use multiple dispatch
class Spam(metaclass=MultipleMeta):
    def bar(self, x:int, y:int):
        print('Bar 1:', x, y)
    def bar(self, s:str, n:int = 0):
        print('Bar 2:', s, n)

# overloaded __init__
class Date(metaclass=MultipleMeta):
    def __init__(self, year: int, month:int, day:int):
        self.year = year
        self.month = month
        self.day = day

    def __init__(self):
        t = time.localtime()
        self.__init__(t.tm_year, t.tm_mon, t.tm_mday)

if '__main__' == __name__:
    s = Spam()
    s.bar(2, 3)
    s.bar('hello')
    s.bar('hello', 5)
    try:
        s.bar(2, 'hello')
    except TypeError as e:
        print(e)

    # overloaded __init__
    d = Date(2012, 12, 21)
    print(d.year, d.month, d.day)
    # get today's date
    e = Date()
    print(e.year, e.month, e.day)

# 用装饰器实现类似功能
class multimethod:
    def __init__(self, func):
        self._methods = {}
        self.__name__ = func.__name__
        self._default = func

    def match(self, *types):
        def register(func):
            ndefaults = len(func.__defaults__) if func.__defaults__ else 0
            for n in range(ndefaults+1):
                self._methods[types[:len(types) - n]] = func
            return self
        return register

    def __call__(self, *args):
        types = tuple(type(arg) for arg in args[1:])
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            return self._default(*args)
        
    def __get__(self, instance, cls):
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

class Spam:
    @multimethod
    def bar(self, *args):
        # default method called if no match
        raise TypeError('No matching method for bar')

    @bar.match(int, int)
    def bar(self, x, y):
        print('Bar 1:', x, y)

    @bar.match(str, int)
    def bar(self, s, n = 0):
        print('Bar 2:', s, n)

if '__main__' == __name__:
    s = Spam()
    s.bar(2, 3)
    s.bar('hello')
    s.bar('hello', 5)
    try:
        s.bar(2, 'hello')
    except TypeError as e:
        print(e)

#21 避免出现重复的属性方法
def typed_property(name, expected_type):
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)
    return prop

class Person:
    name = typed_property('name', str)
    age = typed_property('age', int)
    def __init__(self, name, age):
        self.name = name
        self.age = age

if '__main__' == __name__:
    p = Person('Dave', 39)
    p.name = 'Guido'
    try:
        p.age = 'Old'
    except TypeError as e:
        print(e)

#22 以简单的方式定义上下文管理器
from contextlib import contextmanager

@contextmanager
def timethis(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{}: {}'.format(label, end - start))

with timethis('counting'):
    n = 1000000
    while n > 0:
        n -= 1

@contextmanager
def list_transaction(orig_list):
    working = list(orig_list)
    yield working
    orig_list[:] = working

# Example
if '__main__' == __name__:
    items = [1, 2, 3]
    with list_transaction(items) as working:
        working.append(4)
        working.append(5)
    print(items)
    try:
        with list_transaction(items) as working:
            working.append(6)
            working.append(7)
            raise RuntimeError('oops')
    except RuntimeError as e:
        print(e)

#23 执行带有局部副作用的代码
a = 13
exec('b = a + 1')
print(b)

try:
    def test():
        a1 = 13
        exec('b1 = a1 + 1')
        print(b1)
    test()
except NameError as e:
    print(e)

def test():
    a1 = 13
    loc = locals()
    exec('b1 = a1 + 1')
    b1 = loc['b1']
    print(b1)
test()

def test1():
    a1 = 0
    exec('a1 += 1')
    print(a1)
test1()

def test2():
    a1 = 0
    loc = locals()
    print('before:', loc)
    exec('a1 += 1')
    print('after:', loc)
    print('a1=', a1)
test2()

def test3():
    a1 = 0
    loc = locals()
    print(loc)
    exec('a1 += 1')
    print(loc)
    locals()
    print(loc)
test3()

def test4():
    a1 = 12
    loc = {'a1': a1}
    glb = {}
    exec('b1 = a1 + 1', glb, loc)
    b1 = loc['b1']
    print(b1)
test4()

#24 解析并分析Python源代码
import ast

ex = ast.parse('2 + 3*4 + x', mode='eval')
print(ex)
print(ast.dump(ex))

top = ast.parse('for i in range(10): print(i)', mode='exec')
print(top)
print(ast.dump(top))

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loaded = set()
        self.stored = set()
        self.deleted = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stored.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.deleted.add(node.id)

if '__main__' == __name__:
    # python code
    code = """
for i in range(10):
    print(i)
del i
"""
    # parse into an AST
    top = ast.parse(code, mode='exec')

    # feed the AST to analyze name usage
    c = CodeAnalyzer()
    c.visit(top)
    print('loaded:', c.loaded)
    print('stored:', c.stored)
    print('deleted:', c.deleted)

# Node visitor that lowers globally accessed names into
# the function body as local variables. 
class NameLower(ast.NodeVisitor):
    def __init__(self, lowered_names):
        self.lowered_names = lowered_names

    def visit_FunctionDef(self, node):
        # Compile some assignments to lower the constants
        code = '__globals = globals()\n'
        code += '\n'.join("{0} = __globals['{0}']".format(name)
                          for name in self.lowered_names)

        code_ast = ast.parse(code, mode='exec')
        print(ast.dump(code_ast))

        # Inject new statements into the function body
        node.body[:0] = code_ast.body

        # Save the function object
        self.func = node

# Decorator that turns global names into locals
def lower_names(*namelist):
    def lower(func):
        srclines = inspect.getsource(func).splitlines()
        # Skip source lines prior to the @lower_names decorator
        for n, line in enumerate(srclines):
            if '@lower_names' in line:
                break

        src = '\n'.join(srclines[n+1:])
        # Hack to deal with indented code
        if src.startswith((' ','\t')):
            src = 'if 1:\n' + src
        top = ast.parse(src, mode='exec')

        # Transform the AST 
        cl = NameLower(namelist)
        cl.visit(top)

        # Execute the modified AST
        temp = {}
        exec(compile(top,'','exec'), temp, temp)

        # Pull out the modified code object
        func.__code__ = temp[func.__name__].__code__
        return func
    return lower

INCR = 1

def countdown1(n):
    while n > 0:
        n -= INCR

@lower_names('INCR')
def countdown2(n):
    while n > 0:
        n -= INCR

if '__main__' == __name__:
    import time
    print('Running a performance check')

    start = time.time()
    countdown1(1000000)
    end = time.time()
    print('countdown1:', end - start)

    start = time.time()
    countdown2(1000000)
    end = time.time()
    print('countdown2:', end - start)

#25 将Python源码分解为字节码
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
    print('Blastoff!')

import dis

dis.dis(countdown)
c = countdown.__code__.co_code
print(c)
import opcode
print(opcode.opname[c[0]])
print(opcode.opname[c[3]])

def generate_opcodes(codebytes):
    extended_arg = 0
    i = 0
    n = len(codebytes)
    while i < n:
        op = codebytes[i]
        i += 1
        if op >= opcode.HAVE_ARGUMENT:
            oparg = codebytes[i] + codebytes[i+1]*256 + extended_arg
            extended_arg = 0
            i += 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536
                continue
        else:
            oparg = None
        yield (op, oparg)

for op, oparg in generate_opcodes(countdown.__code__.co_code):
    print(op, opcode.opname[op], oparg)
