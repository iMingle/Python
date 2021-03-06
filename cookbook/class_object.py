"""Classes and Objects.

"""


# 1 修改实例的字符串表示
class Pair:
    def __init__(self, x, y):
        super(Pair, self).__init__()
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Pair({0.x!r}, {0.y!r})'.format(self)

    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)


p = Pair(3, 4)
print('p is {0!r}'.format(p))
print('p is {0}'.format(p))

# 2 自定义字符串的输出格式
_formats = {
    'ymd': '{d.year}-{d.month}-{d.day}',
    'mdy': '{d.month}/{d.day}/{d.year}',
    'dmy': '{d.day}/{d.month}/{d.year}'
}


class Date:
    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, code):
        if '' == code:
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)


d = Date(2016, 7, 31)
print(format(d))
print(format(d, 'mdy'))
print('the date is {:ymd}'.format(d))
print('the date is {:mdy}'.format(d))

from datetime import date


d = date(2016, 8, 1)
print(format(d))
print(format(d, '%A, %B %d, %Y'))
print('the end is {:%d %b %Y}'.format(d))

# 3 让对象支持上下文管理协议
# 要让对象能够兼容with语句,需要实现__enter__()和__exit__()方法
from socket import socket, AF_INET, SOCK_STREAM


class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        super(LazyConnection, self).__init__()
        self.address = address
        self.family = AF_INET
        self.type = SOCK_STREAM
        self.connections = []

    def __enter__(self):
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        self.connections.append(sock)
        return sock

    def __exit__(self, exc_ty, exc_val, tb):
        self.connections.pop().close()


from functools import partial


conn = LazyConnection(('www.python.org', 80))
with conn as s:
    # conn.__enter__() executes: connection open
    s.send(b'GET /index.html HTTP/1.0\r\n')
    s.send(b'Host: www.python.org\r\n')
    s.send(b'\r\n')
    resp = b''.join(iter(partial(s.recv, 8192), b''))
    # conn.__exit__() executes: connection closed
    with conn as s1:
        s1.send(b'GET /index.html HTTP/1.0\r\n')
        s1.send(b'Host: www.python.org\r\n')
        s1.send(b'\r\n')
        resp = b''.join(iter(partial(s1.recv, 8192), b''))


# 4 当创建大量实例时如何节省内存
class Date:
    # 当定义了__slots__属性时,Python就会针对实例采用一种更加紧凑的内部表示
    __slots__ = ['year', 'month', 'day']

    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day


# 5 将名称封装到类中
# _开头的名字应该总是被认为只属于内部实现
class A:
    def __init__(self):
        super(A, self).__init__()
        self._internal = 0
        self.public = 1

    def public_method(self):
        pass

    def _internal_method(self):
        pass


class B:
    def __init__(self):
        super(B, self).__init__()
        self.__private = 0

    def __private_method(self):
        pass

    def public_method(self):
        self.__private_method()


class C(B):
    def __init__(self, arg):
        super(C, self).__init__()
        self.__private = 1  # 不会覆盖B.__private

    # 不会覆盖B.__private_method()
    def __private_method(self):
        pass

    # 6 创建可管理的属性


class Person:
    def __init__(self, first_name):
        super(Person, self).__init__()
        self.first_name = first_name

    # getter function
    @property
    def first_name(self):
        return self._first_name

    # setter function
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('expected a string')
        self._first_name = value

    # deleter function (optional)
    @first_name.deleter
    def first_name(self):
        raise AttributeError('cannot delete attribute')


a = Person('Mingle')
print(a.first_name)  # calls the getter
a.first_name = 'Jack'  # calls the setter
print(a.first_name)
# del a.first_name

print(Person.first_name.fget)
print(Person.first_name.fset)
print(Person.first_name.fdel)


# 对于已经存在的get和set方法,同样可以将它们定义为property
class Person:
    def __init__(self, first_name):
        super(Person, self).__init__()
        self.set_first_name(first_name)

    # getter function
    def get_first_name(self):
        return self._first_name

    # setter function
    def set_first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('expected a string')
        self._first_name = value

    # deleter function (optional)
    def del_first_name(self):
        raise AttributeError('cannot delete attribute')


a = Person('Mingle1')
print(a.get_first_name())  # calls the getter
a.set_first_name('Jack1')  # calls the setter
print(a.get_first_name())
# a.del_first_name()

import math


class Circle:
    def __init__(self, radius):
        super(Circle, self).__init__()
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius


c = Circle(4.0)
print(c.radius)
print(c.area)
print(c.perimeter)


# 7 调用父类中的方法
class A:
    def __init__(self):
        self.x = 0

    def spam(self):
        print('A.spam')


class B(A):
    def __init__(self):
        super(B, self).__init__()
        self.y = 1

    def spam(self):
        print('B.spam')
        super().spam()  # call parent spam()


class Proxy:
    def __init__(self, obj):
        super(Proxy, self).__init__()
        self._obj = obj

    # delegate attribute lookup to internal obj
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # delegate attribute assignment
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)  # call original __setattr__
        else:
            setattr(self._obj, name, value)


class Base:
    def __init__(self):
        print('Base.__init__')


class A(Base):
    def __init__(self):
        Base.__init__(self)
        print('A.__init__')


class B(Base):
    def __init__(self):
        Base.__init__(self)
        print('B.__init__')


class C(A, B):
    def __init__(self):
        A.__init__(self)
        B.__init__(self)
        print('C.__init__')


c = C()  # B.__init__()调用2次


class Base:
    def __init__(self):
        print('Base.__init__')


class A(Base):
    def __init__(self):
        super(A, self).__init__()
        print('A.__init__')


class B(Base):
    def __init__(self):
        super(B, self).__init__()
        print('B.__init__')


class C(A, B):
    def __init__(self):
        super(C, self).__init__()
        print('C.__init__')


c = C()
# 方法解析顺序(MRO)
# 1. 先检查子类再检查父类
# 2. 有多个父类时,按照MRO列表的顺序依次检查
# 3. 如果下一个待选的类出现了两个合法的选择,那么就从第一个父类中选取
print(C.__mro__)


# 8 在子类中扩展属性
class Person:
    def __init__(self, name):
        super(Person, self).__init__()
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('excepted a string')
        self._name = value

    @name.deleter
    def name(self):
        raise AttributeError('cannot delete attribute')


class SubPerson(Person):
    @property
    def name(self):
        print('getting name')
        return super().name

    @name.setter
    def name(self, value):
        print('setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print('deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)


s = SubPerson('Mingle')
print(s.name)
s.name = 'Page'


# s.name = 42 # error

# 只想扩展属性中的一个方法
class SubPerson(Person):
    @Person.name.setter
    def name(self, value):
        print('setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)


# 扩展描述符
class String:
    def __init__(self, name):
        super(String, self).__init__()
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError('excepted a string')
        instance.__dict__[self.name] = value


class Person:
    name = String('name')

    def __init__(self, name):
        super(Person, self).__init__()
        self.name = name


# extending a descriptor with a property
class SubPerson(Person):
    @property
    def name(self):
        print('getting name')
        return super().name

    @name.setter
    def name(self, value):
        print('setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print('deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)


# 9 创建一种新形式的类属性或实例属性
# descriptor attribute for an integer type-checked attribute
# 描述符只能在类的层次上定义,不能根据实例来产生
class Integer:
    def __init__(self, name):
        super(Integer, self).__init__()
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('expected an int')
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class Point:
    x = Integer('x')
    y = Integer('y')

    def __init__(self, x, y):
        super(Point, self).__init__()
        self.x = x
        self.y = y


p = Point(2, 4)
print(p.x)  # calls Point.x.__get__(p, Point)
print(p.y)  # calls Point.y.__get__(p, Point)
p.y = 5  # calls Point.y.__set__(p, 5)
p.x = 2  # calls Point.x.__set__(p, 2)
print(p.x)
print(p.y)


# descriptor for a type-checked attribute
class Typed:
    def __init__(self, name, expected_type):
        super(Typed, self).__init__()
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


# class decorator that applies it to selected attributes
def typeassert(**kwargs):
    def decorate(cls):
        for name, expected_type in kwargs.items():
            # attach a Typed descriptor to the class
            setattr(cls, name, Typed(name, expected_type))
        return cls

    return decorate


@typeassert(name=str, shares=int, price=float)
class Stock:
    def __init__(self, name, shares, price):
        super(Stock, self).__init__()
        self.name = name
        self.shares = shares
        self.price = price


s = Stock('GOOG', 100, 789.65)
print(s.name)
print(s.shares)
print(s.price)


# 10 让属性具有惰性求值的能力
# 只有当被访问的属性不在底层的实例字典中时,__get__()方法才会得到调用
class lazyproperty:
    """延迟属性描述器"""

    def __init__(self, func):
        super(lazyproperty, self).__init__()
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Circle:
    def __init__(self, radius):
        super(Circle, self).__init__()
        self.radius = radius

    @lazyproperty
    def area(self):
        print('computing area')
        return math.pi * self.radius ** 2

    @lazyproperty
    def perimeter(self):
        print('computing perimeter')
        return 2 * math.pi * self.radius


c = Circle(4.0)
print(c.radius)
print(c.area)
print(c.area)
print(c.perimeter)
print(c.perimeter)

c = Circle(4.0)
print(vars(c))
print(Circle.__dict__)
print(c.__dict__)
print(c.area)
print(vars(c))
print(c.area)
del c.area
print(vars(c))
print(c.area)
# 缺点,值可变
c.area = 25
print(c.area)


# 不可变实现
def lazyproperty(func):
    """延迟属性描述器,属性不可变,但执行效率会稍打折扣"""
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


class Circle:
    def __init__(self, radius):
        super(Circle, self).__init__()
        self.radius = radius

    @lazyproperty
    def area(self):
        print('computing area')
        return math.pi * self.radius ** 2

    @lazyproperty
    def perimeter(self):
        print('computing perimeter')
        return 2 * math.pi * self.radius


c = Circle(4.0)
print(c.area)
print(c.area)


# c.area = 25 # AttributeError: can't set attribute

# 11 简化数据结构的初始化过程
class Structure:
    # class variable that specifies expected fields
    _fields = []

    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('expected {} arguments'.format(len(self._fields)))
        # set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)


if '__main__' == __name__:
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    class Point(Structure):
        _fields = ['x', 'y']


    class Circle(Structure):
        _fields = ['radius']

        def area(self):
            return math.pi * self.radius ** 2


    s = Stock('ACME', 50, 90.1)
    p = Point(2, 3)
    c = Circle(4.5)
    try:
        s2 = Stock('ACME', 50)
    except TypeError as e:
        print(e)


# 支持关键字映射
class Structure:
    # class variable that specifies expected fields
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('expected {} arguments'.format(len(self._fields)))
        # set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        # set the remaining keyword arguments
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))
        # check for any remaining unknown arguments
        if kwargs:
            raise TypeError('invalid arguments(s): {}'.format(','.join(kwargs)))


if '__main__' == __name__:
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    s1 = Stock('ACME', 50, 90.1)
    s2 = Stock('ACME', 50, price=90.1)
    s3 = Stock('ACME', shares=50, price=90.1)


# 利用关键字参数来给类添加额外的属性
class Structure:
    # class variable that specifies expected fields
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # set the additional arguments (if any)
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))

        if kwargs:
            raise TypeError('duplicate values for {}'.format(','.join(kwargs)))


if '__main__' == __name__:
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, 91.1, date='8/2/2012')

# 12 定义一个接口或抽象基类
# 要定义一个抽象基类,可以使用abc模块
from abc import ABCMeta, abstractmethod


class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass

    @abstractmethod
    def write(self, data):
        pass


# a = IStream() # TypeError: Can't instantiate abstract class IStream with abstract methods read, write

class SocketStream(IStream):
    def read(self, maxbytes=-1):
        pass

    def write(self, data):
        pass


# 检查接口
def serialize(obj, stream):
    if not isinstance(stream, IStream):
        raise TypeError('expected an IStream')


# 抽象基类允许其他的类向其注册,然后实现所需的接口
import io


# register the built-in I/O classes as supporting our interface
IStream.register(io.IOBase)
# open a normal file and type check
f = open('data/somefile.txt')
print(isinstance(f, IStream))


# @abstractmethod同样可以施加到静态方法,类方法和property属性上,
# 只要确保以合适的顺序进行添加即可.
class A(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, value):
        pass

    @classmethod
    @abstractmethod
    def method1(cls):
        pass

    @staticmethod
    @abstractmethod
    def method2():
        pass

    def method3(self):
        print('method3')


class B(A):
    def name(self):
        print('name')

    def name(self, value):
        print('name.setter')

    def method1(cls):
        print('method1')

    def method2():
        print('method2')


b = B()
b.method3()

import collections


x = [1, 2, 3, 4]
if isinstance(x, collections.Sequence):
    print('x is collections.Sequence')
if isinstance(x, collections.Iterable):
    print('x is collections.Iterable')
if isinstance(x, collections.Sized):
    print('x is collections.Sized')
if isinstance(x, collections.Mapping):
    print('x is collections.Mapping')

from decimal import Decimal
import numbers


x = Decimal('3.4')
print(isinstance(x, numbers.Real))  # False


# 13 实现一种数据模型或类型系统
class Descriptor(object):
    """base class. uses a descriptor to set a value"""

    def __init__(self, name=None, **opts):
        super(Descriptor, self).__init__()
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class Typed(Descriptor):
    """descriptor for enforcing types"""
    expected_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('expected ' + str(self.expected_type))
        super().__set__(instance, value)


class Unsigned(Descriptor):
    """descriptor for enforcing values"""

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('expected >= 0')
        super().__set__(instance, value)


class MaxSized(Descriptor):
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super().__set__(instance, value)


class Integer(Typed):
    expected_type = int


class UnsignedInteger(Integer, Unsigned):
    pass


class Float(Typed):
    expected_type = float


class UnsignedFloat(Float, Unsigned):
    pass


class String(Typed):
    expected_type = str


class SizedString(String, MaxSized):
    pass


def test(s):
    """testing code"""
    print(s.name)
    s.shares = 75
    print(s.shares)
    try:
        s.shares = -10
    except ValueError as e:
        print(e)
    try:
        s.price = 'a lot'
    except TypeError as e:
        print(e)

    try:
        s.name = 'ABRACADABRA'
    except ValueError as e:
        print(e)


if '__main__' == __name__:
    class Stock:
        # specify constraints
        name = SizedString('name', size=8)
        shares = UnsignedInteger('shares')
        price = UnsignedFloat('price')

        def __init__(self, name, shares, price):
            super(Stock, self).__init__()
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 90.1)
    test(s)


# 简化在类中设定约束的步骤
def check_attributes(**kwargs):
    """class decorator to apply constraints"""

    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls

    return decorate


if '__main__' == __name__:
    @check_attributes(name=SizedString(size=8),
                      shares=UnsignedInteger,
                      price=UnsignedFloat)
    class Stock:
        def __init__(self, name, shares, price):
            super(Stock, self).__init__()
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 90.1)
    test(s)


# 使用元类检查
class checkedmeta(type):
    def __new__(cls, clsname, bases, methods):
        # attach attribute names to the descriptors
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)


if '__main__' == __name__:
    class Stock(metaclass=checkedmeta):
        name = SizedString(size=8)
        shares = UnsignedInteger()
        price = UnsignedFloat()

        def __init__(self, name, shares, price):
            super(Stock, self).__init__()
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 90.1)
    test(s)


# 使用类装饰器的备选方案,执行速度快
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        self.__dict__.update(opts)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


def Typed(expected_type, cls=None):
    if cls is None:
        return lambda cls: Typed(expected_type, cls)

    super_set = cls.__set__

    def __set__(self, instance, value):
        if not isinstance(value, expected_type):
            raise TypeError('expected ' + str(expected_type))
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


def Unsigned(cls):
    super_set = cls.__set__

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('expected >= 0')
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


def MaxSized(cls):
    super_init = cls.__init__

    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        self.size = opts['size']
        super_init(self, name, **opts)

    cls.__init__ = __init__

    super_set = cls.__set__

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


@Typed(int)
class Integer(Descriptor):
    pass


@Unsigned
class UnsignedInteger(Integer):
    pass


@Typed(float)
class Float(Descriptor):
    pass


@Unsigned
class UnsignedFloat(Float):
    pass


@Typed(str)
class String(Descriptor):
    pass


@MaxSized
class SizedString(String):
    pass


def check_attributes(**kwargs):
    """class decorator to apply constraints"""

    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls

    return decorate


class checkedmeta(type):
    """a metaclass that applies checking"""

    def __new__(cls, clsname, bases, methods):
        # attach attribute names to the descriptors
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)


if __name__ == '__main__':
    print('# --- Class with descriptors')


    class Stock:
        # Specify constraints
        name = SizedString('name', size=8)
        shares = UnsignedInteger('shares')
        price = UnsignedFloat('price')

        def __init__(self, name, shares, price):
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 91.1)
    test(s)

    print('# --- Class with class decorator')


    @check_attributes(name=SizedString(size=8),
                      shares=UnsignedInteger,
                      price=UnsignedFloat)
    class Stock:
        def __init__(self, name, shares, price):
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 91.1)
    test(s)

    print('# --- Class with metaclass')


    class Stock(metaclass=checkedmeta):
        name = SizedString(size=8)
        shares = UnsignedInteger()
        price = UnsignedFloat()

        def __init__(self, name, shares, price):
            self.name = name
            self.shares = shares
            self.price = price


    s = Stock('ACME', 50, 91.1)
    test(s)

# 14 实现自定义的容器
import collections


class A(collections.Iterable):
    def __iter__(self):
        pass


a = A()

# collections.Sequence() # TypeError: Can't instantiate abstract class Sequence 
# with abstract methods __getitem__, __len__

import bisect


class SortedItems(collections.Sequence):
    """排序列表"""

    def __init__(self, initial=None):
        self._items = sorted(initial) if initial is not None else []

    # required sequence methods
    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    # method for adding an item in the right location
    def add(self, item):
        bisect.insort(self._items, item)


items = SortedItems([5, 1, 3])
print(list(items))
print(items[0])
print(items[-1])
items.add(2)
print(list(items))
items.add(-10)
print(list(items))
print(items[1:4])
print(3 in items)
print(len(items))
for n in items:
    print(n)


class Items(collections.MutableSequence):
    def __init__(self, initial=None):
        self._items = list(initial) if initial is not None else []

    # required sequence methods
    def __getitem__(self, index):
        print('Getting:', index)
        return self._items[index]

    def __setitem__(self, index, value):
        print('Setting:', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        print('Deleting:', index)
        del self._items[index]

    def insert(self, index, value):
        print('Inserting:', index, value)
        self._items.insert(index, value)

    def __len__(self):
        print('Len')
        return len(self._items)


if '__main__' == __name__:
    a = Items([1, 2, 3])
    print(len(a))
    a.append(4)
    a.append(2)
    print(a.count(2))
    a.remove(3)


# 15 委托属性的访问
class A:
    def spam(self, x):
        print('A.spam')

    def foo(self):
        print('A.foo')


class B:
    def __init__(self):
        self._a = A()

    def bar(self):
        print('B.bar')

    # expose all of the methods defined on class A   
    def __getattr__(self, name):
        return getattr(self._a, name)


if '__main__' == __name__:
    b = B()
    b.bar()
    b.spam(42)


class Proxy:
    """a proxy class that wraps around another object, but exposes its public attributes"""

    def __init__(self, obj):
        self._obj = obj

    # delegate attribute lookup to internal obj
    def __getattr__(self, name):
        print('getattr:', name)
        return getattr(self._obj, name)

    # delegate attribute assignment
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print('setattr:', name, value)
            setattr(self._obj, name, value)

    # delegate attribute deletion
    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            print('delattr:', name)
            delattr(self._obj, name)


if '__main__' == __name__:
    class Spam:
        def __init__(self, x):
            self.x = x

        def bar(self, y):
            print('Spam.bar:', self.x, y)


    # create an instance
    s = Spam(2)
    # create a proxy around it
    p = Proxy(s)
    # access the proxy
    print(p.x)
    p.bar(3)
    p.x = 37


class ListLike:
    def __init__(self):
        self._items = []

    def __getattr__(self, name):
        return getattr(self._items, name)

    # added special methods to support certain list operations
    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]


if '__main__' == __name__:
    a = ListLike()
    a.append(2)
    a.insert(0, 1)
    a.sort()
    print(len(a))
    print(a[0])

# 16 在类中定义多个构造函数
import time


class Date:
    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday)


a = Date(2016, 8, 6)
b = Date.today()
print(a)
print(b)


class NewDate(Date):
    pass


c = Date.today()
d = NewDate.today()
print(c)  # Date
print(d)  # NewDate


# 17 不通过调用init来创建实例
class Date:
    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day


d = Date.__new__(Date)
print(d)
# print(d.year) # AttributeError: 'Date' object has no attribute 'year'
data = {'year': 2016, 'month': 8, 'day': 6}
for key, value in data.items():
    setattr(d, key, value)
print(d.year)

from time import localtime


class Date:
    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        d = cls.__new__(cls)
        t = localtime()
        d.year = t.tm_year
        d.month = t.tm_mon
        d.day = t.tm_mday
        return d


# 18 用Mixin技术来扩展类定义
class LoggedMappingMixin:
    """add logging to get/set/delete operations for debugging."""
    __slots__ = ()

    def __getitem__(self, key):
        print('getting ' + str(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        print('setting {} = {!r}'.format(key, value))
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        print('deleting ' + str(key))
        return super().__delitem__(key)


class SetOnceMappingMixin:
    """only allow a key to be set once."""
    __slots__ = ()

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(str(key) + ' already set')
        return super().__setitem__(key, value)


class StringKeysMappingMixin:
    """restrict keys to strings only."""
    __slots__ = ()

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('keys must be strings')
        return super().__setitem__(key, value)


class LoggedDict(LoggedMappingMixin, dict):
    pass


d = LoggedDict()
d['x'] = 23
print(d['x'])
del d['x']

from collections import defaultdict


class SetOnceDefaultDict(SetOnceMappingMixin, defaultdict):
    pass


d = SetOnceDefaultDict(list)
d['x'].append(2)
d['y'].append(3)
d['x'].append(10)
try:
    d['x'] = 23
except KeyError as e:
    print(e)

from collections import OrderedDict


class StringOrderedDict(StringKeysMappingMixin,
                        SetOnceMappingMixin,
                        OrderedDict):
    pass


d = StringOrderedDict()
d['x'] = 23
try:
    d[42] = 10
except TypeError as e:
    print(e)

try:
    d['x'] = 42
except KeyError as e:
    print(e)


class RestrictKeysMixin:
    def __init__(self, *args, _restrict_key_type, **kwargs):
        self.__restrict_key_type = _restrict_key_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(key, self.__restrict_key_type):
            raise TypeError('Keys must be ' + str(self.__restrict_key_type))
        super().__setitem__(key, value)


class RDict(RestrictKeysMixin, dict):
    pass


d = RDict(_restrict_key_type=str)
e = RDict([('name', 'Dave'), ('n', 37)], _restrict_key_type=str)
f = RDict(name='Dave', n=37, _restrict_key_type=str)
print(f)
try:
    f[42] = 10
except TypeError as e:
    print(e)


# 用类装饰器实现mixin
def LoggedMapping(cls):
    cls_getitem = cls.__getitem__
    cls_setitem = cls.__setitem__
    cls_delitem = cls.__delitem__

    def __getitem__(self, key):
        print('getting %s' % key)
        return cls_getitem(self, key)

    def __setitem__(self, key, value):
        print('setting %s = %r' % (key, value))
        return cls_setitem(self, key, value)

    def __delitem__(self, key):
        print('deleting %s' % key)
        return cls_delitem(self, key)

    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__
    cls.__delitem__ = __delitem__
    return cls


@LoggedMapping
class LoggedDict(dict):
    pass


d = LoggedDict()
d['x'] = 23
print(d['x'])
del d['x']


# 19 实现带有状态的对象或状态机
class Connection:
    def __init__(self):
        self.new_state(ClosedConnectionState)

    def new_state(self, newstate):
        self._state = newstate

    # delegate to the state class
    def read(self):
        return self._state.read(self)

    def write(self, data):
        return self._state.write(self, data)

    def open(self):
        return self._state.open(self)

    def close(self):
        return self._state.close(self)


class ConnectionState:
    """connection state base class."""

    @staticmethod
    def read(conn):
        raise NotImplementedError()

    @staticmethod
    def write(conn, data):
        raise NotImplementedError()

    @staticmethod
    def open(conn):
        raise NotImplementedError()

    @staticmethod
    def close(conn):
        raise NotImplementedError()


# Implementation of different states
class ClosedConnectionState(ConnectionState):
    @staticmethod
    def read(conn):
        raise RuntimeError('not open')

    @staticmethod
    def write(conn, data):
        raise RuntimeError('not open')

    @staticmethod
    def open(conn):
        conn.new_state(OpenConnectionState)

    @staticmethod
    def close(conn):
        raise RuntimeError('already closed')


class OpenConnectionState(ConnectionState):
    @staticmethod
    def read(conn):
        print('reading')

    @staticmethod
    def write(conn, data):
        print('writing')

    @staticmethod
    def open(conn):
        raise RuntimeError('already open')

    @staticmethod
    def close(conn):
        conn.new_state(ClosedConnectionState)


if '__main__' == __name__:
    c = Connection()
    print(c)
    try:
        c.read()
    except RuntimeError as e:
        print(e)

    c.open()
    print(c)
    c.read()
    c.close()
    print(c)


class Connection:
    def __init__(self):
        self.new_state(ClosedConnection)

    def new_state(self, state):
        self.__class__ = state

    def read(self):
        raise NotImplementedError()

    def write(self, data):
        raise NotImplementedError()

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ClosedConnection(Connection):
    def read(self):
        raise RuntimeError('not open')

    def write(self, data):
        raise RuntimeError('not open')

    def open(self):
        self.new_state(OpenConnection)

    def close(self):
        raise RuntimeError('already closed')


class OpenConnection(Connection):
    def read(self):
        print('reading')

    def write(self, data):
        print('writing')

    def open(self):
        raise RuntimeError('already open')

    def close(self):
        self.new_state(ClosedConnection)


if '__main__' == __name__:
    c = Connection()
    print(c)
    try:
        c.read()
    except RuntimeError as e:
        print(e)

    c.open()
    print(c)
    c.read()
    c.close()
    print(c)


# 20 调用对象上的方法,方法名以字符串形式输出
class Point:
    def __init__(self, x, y):
        super(Point, self).__init__()
        self.x = x
        self.y = y

    def __repr(self):
        return 'Point({!r:}, {!r:})'.format(self.x, self.y)

    def distance(self, x, y):
        return math.hypot(self.x - x, self.y - y)


p = Point(2, 3)
d = getattr(p, 'distance')(0, 0)
print(d)

import operator


print(operator.methodcaller('distance', 0, 0)(p))

points = [
    Point(1, 2),
    Point(3, 0),
    Point(10, -3),
    Point(-5, -7),
    Point(-1, 8),
    Point(3, 2)
]

# sort by distance from origin (0, 0)
points.sort(key=operator.methodcaller('distance', 0, 0))
for p in points:
    print(p)


# 21 实现访问者模式
class Node:
    pass


class UnaryOperator(Node):
    def __init__(self, operand):
        self.operand = operand


class BinaryOperator(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Add(BinaryOperator):
    pass


class Sub(BinaryOperator):
    pass


class Mul(BinaryOperator):
    pass


class Div(BinaryOperator):
    pass


class Negate(UnaryOperator):
    pass


class Number(Node):
    def __init__(self, value):
        self.value = value


# representation of 1 + 2 * (3 - 4) / 5
t1 = Sub(Number(3), Number(4))
t2 = Mul(Number(2), t1)
t3 = Div(t2, Number(5))
t4 = Add(Number(1), t3)


# 访问者模式
class NodeVisitor:
    def visit(self, node):
        methodname = 'visit_' + type(node).__name__
        method = getattr(self, methodname, None)
        if method is None:
            method = self.generic_visit
        return method(node)

    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))


class Evaluator(NodeVisitor):
    """计算器"""

    def visit_Number(self, node):
        return node.value

    def visit_Add(self, node):
        return self.visit(node.left) + self.visit(node.right)

    def visit_Sub(self, node):
        return self.visit(node.left) - self.visit(node.right)

    def visit_Mul(self, node):
        return self.visit(node.left) * self.visit(node.right)

    def visit_Div(self, node):
        return self.visit(node.left) / self.visit(node.right)

    def visit_Negate(self, node):
        return -node.operand


e = Evaluator()
print(e.visit(t4))


class StackCode(NodeVisitor):
    """generate stack instructions."""

    def generate_code(self, node):
        self.instructions = []
        self.visit(node)
        return self.instructions

    def visit_Number(self, node):
        self.instructions.append(('PUSH', node.value))

    def binop(self, node, instruction):
        self.visit(node.left)
        self.visit(node.right)
        self.instructions.append((instruction,))

    def visit_Add(self, node):
        self.binop(node, 'ADD')

    def visit_Sub(self, node):
        self.binop(node, 'SUB')

    def visit_Mul(self, node):
        self.binop(node, 'MUL')

    def visit_Div(self, node):
        self.binop(node, 'DIV')

    def unaryop(self, node, instruction):
        self.visit(node.operand)
        self.instructions.append((instruction,))

    def visit_Negate(self, node):
        self.unaryop(node, 'NEG')


s = StackCode()
print(s.generate_code(t4))

# 22 实现非递归的访问者模式
import types


class NodeVisitor:
    def visit(self, node):
        stack = [node]
        last_result = None
        while stack:
            try:
                last = stack[-1]
                if isinstance(last, types.GeneratorType):
                    stack.append(last.send(last_result))
                    last_result = None
                elif isinstance(last, Node):
                    stack.append(self._visit(stack.pop()))
                else:
                    last_result = stack.pop()
            except StopIteration:
                stack.pop()
        return last_result

    def _visit(self, node):
        methodname = 'visit_' + type(node).__name__
        method = getattr(self, methodname, None)
        if method is None:
            method = self.generic_visit
        return method(node)

    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))


if '__main__' == __name__:
    a = Number(0)
    for n in range(1, 500):
        a = Add(a, Number(n))
    e = Evaluator()
    try:
        print(e.visit(a))
    except RuntimeError as e:
        print(e)


# 非递归形式
class Evaluator(NodeVisitor):
    """计算器"""

    def visit_Number(self, node):
        return node.value

    def visit_Add(self, node):
        yield (yield node.left) + (yield node.right)

    def visit_Sub(self, node):
        yield (yield node.left) - (yield node.right)

    def visit_Mul(self, node):
        yield (yield node.left) * (yield node.right)

    def visit_Div(self, node):
        yield (yield node.left) / (yield node.right)

    def visit_Negate(self, node):
        yield -(yield node.operand)


if '__main__' == __name__:
    a = Number(0)
    for n in range(1, 1000):
        a = Add(a, Number(n))
    e = Evaluator()
    try:
        print(e.visit(a))
    except RuntimeError as e:
        print(e)

# 23 在环状数据结构中管理内存
import weakref


class Node:
    """树节点"""

    def __init__(self, value):
        super(Node, self).__init__()
        self.value = value
        self._parent = None
        self.children = []

    def __repr__(self):
        return 'Node({!r:})'.format(self.value)

    # property that manages the parent as a weak-reference
    @property
    def parent(self):
        return self._parent if self._parent is not None else self._parent()

    @parent.setter
    def parent(self, node):
        self._parent = weakref.ref(node)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


if '__main__' == __name__:
    root = Node('parent')
    c1 = Node('c1')
    c2 = Node('c2')
    root.add_child(c1)
    root.add_child(c2)

    print(c1.parent)
    del root
    print(c1.parent)


class Data:
    """class just to illustrate when deletion occurs"""

    def __del__(self):
        print('Data.__del__')


class Node:
    """树节点"""

    def __init__(self):
        self.data = Data()
        self._parent = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


if '__main__' == __name__:
    a = Data()
    del a
    a = Node()
    del a
    print('delete tree node')
    a = Node()
    a.add_child(Node())
    del a

import gc


gc.collect()


class Node:
    """树节点"""

    def __init__(self):
        self.data = Data()
        self._parent = None
        self.children = []

    # never define like this
    # only here to illustrate pathological behavior
    def __del__(self):
        del self.data
        del self.parent
        del self.children

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


if '__main__' == __name__:
    a = Node()
    a.add_child(Node())
    del a
    gc.collect()
    a = Node()
    a_ref = weakref.ref(a)
    print(a_ref)
    print(a_ref())
    del a
    print(a_ref())

# 24 让类支持比较操作
from functools import total_ordering


class Room:
    def __init__(self, name, length, width):
        self.name = name
        self.length = length
        self.width = width
        self.square_feet = self.length * self.width


# 只要定义__eq__和__lt__,利用@total_ordering就能让其他所有的比较操作正常工作
@total_ordering
class House:
    def __init__(self, name, style):
        self.name = name
        self.style = style
        self.rooms = list()

    @property
    def living_space_footage(self):
        return sum(r.square_feet for r in self.rooms)

    def add_room(self, room):
        self.rooms.append(room)

    def __str__(self):
        return '{}: {} square foot {}'.format(self.name, self.living_space_footage, self.style)

    def __eq__(self, other):
        return self.living_space_footage == other.living_space_footage

    def __lt__(self, other):
        return self.living_space_footage < other.living_space_footage


# Build a few houses, and add rooms to them.
h1 = House('h1', 'Cape')
h1.add_room(Room('Master Bedroom', 14, 21))
h1.add_room(Room('Living Room', 18, 20))
h1.add_room(Room('Kitchen', 12, 16))
h1.add_room(Room('Office', 12, 12))

h2 = House('h2', 'Ranch')
h2.add_room(Room('Master Bedroom', 14, 21))
h2.add_room(Room('Living Room', 18, 20))
h2.add_room(Room('Kitchen', 12, 16))

h3 = House('h3', 'Split')
h3.add_room(Room('Master Bedroom', 14, 21))
h3.add_room(Room('Living Room', 18, 20))
h3.add_room(Room('Office', 12, 16))
h3.add_room(Room('Kitchen', 15, 17))
houses = [h1, h2, h3]

print('Is h1 bigger than h2?', h1 > h2)  # True
print('Is h2 smaller than h3?', h2 < h3)  # True
print('Is h2 greater than or equal to h1?', h2 >= h1)  # False
print('Which one is biggest?', max(houses))  # 'h3: 1101 square foot Split'
print('Which is smallest?', min(houses))  # 'h2: 846 square foot Ranch'

# 25 创建缓存实例
import logging


a = logging.getLogger('foo')
a = logging.getLogger('bar')
print(a is b)  # False
c = logging.getLogger('foo')
print(a is c)  # False


class Spam:
    def __init__(self, name):
        self.name = name


# caching support
_spam_cache = weakref.WeakValueDictionary()


def get_spam(name):
    if name not in _spam_cache:
        s = Spam(name)
        _spam_cache[name] = s
    else:
        s = _spam_cache[name]
    return s


a = get_spam('foo')
b = get_spam('bar')
print(a is b)
c = get_spam('foo')
print(a is c)


class Spam:
    _spam_cache = weakref.WeakValueDictionary()

    # 每次都调用
    def __init__(self, name):
        print('initializing spam')
        self.name = name

    def __new__(cls, name):
        if name in cls._spam_cache:
            return cls._spam_cache[name]
        else:
            self = super().__new__(cls)
            cls._spam_cache[name] = self
            return self


# 重复初始化
s = Spam('dave')
t = Spam('dave')
print(s is t)


# 高级实现技术
class CachedSpamManager:
    """缓存管理器"""

    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get_spam(self, name):
        if name not in self._cache:
            s = Spam(name)
            self._cache[name] = s
        else:
            s = self._cache[name]
        return s

    def clear(self):
        self._cache.clear()


class Spam:
    manager = CachedSpamManager()

    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_spam(name):
        return Spam.manager.get_spam(name)


s = Spam.get_spam('dave')
t = Spam.get_spam('dave')
print(s is t)


class Spam:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("can't instantiate directly")

    @classmethod
    def _new(cls, name):
        self = cls.__new__(cls)
        self.name = name


class CachedSpamManager:
    """缓存管理器"""

    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get_spam(self, name):
        if name not in self._cache:
            s = Spam._new(name)
            self._cache[name] = s
        else:
            s = self._cache[name]
        return s

    def clear(self):
        self._cache.clear()


s = Spam._new('dave')
t = Spam._new('dave')
print(s is t)
