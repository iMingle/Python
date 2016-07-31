"""Classes and Objects.

"""

#1 修改实例的字符串表示
class Pair:
    def __init__(self, x, y):
        super(Pair, self).__init__()
        self.x = x
        self.y = y

    def __repr__(self):
        return "Pair({0.x!r}, {0.y!r})".format(self)

    def __str__(self):
        return "({0.x!s}, {0.y!s})".format(self)
p = Pair(3, 4)
print("p is {0!r}".format(p))
print("p is {0}".format(p))

#2 自定义字符串的输出格式
_formats = {
    "ymd": "{d.year}-{d.month}-{d.day}",
    "mdy": "{d.month}/{d.day}/{d.year}",
    "dmy": "{d.day}/{d.month}/{d.year}"
}

class Date:
    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, code):
        if "" == code:
            code = "ymd"
        fmt = _formats[code]
        return fmt.format(d=self)
d = Date(2016, 7, 31)
print(format(d))
print(format(d, "mdy"))
print("the date is {:ymd}".format(d))
print("the date is {:mdy}".format(d))

from datetime import date
d = date(2016, 8, 1)
print(format(d))
print(format(d, "%A, %B %d, %Y"))
print("the end is {:%d %b %Y}".format(d))

#3 让对象支持上下文管理协议
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

conn = LazyConnection(("www.python.org", 80))
with conn as s:
    # conn.__enter__() executes: connection open
    s.send(b"GET /index.html HTTP/1.0\r\n")
    s.send(b"Host: www.python.org\r\n")
    s.send(b"\r\n")
    resp = b"".join(iter(partial(s.recv, 8192), b""))
    # conn.__exit__() executes: connection closed
    with conn as s1:
        s1.send(b"GET /index.html HTTP/1.0\r\n")
        s1.send(b"Host: www.python.org\r\n")
        s1.send(b"\r\n")
        resp = b"".join(iter(partial(s1.recv, 8192), b""))

#4 当创建大量实例时如何节省内存
class Date:
    # 当定义了__slots__属性时,Python就会针对实例采用一种更加紧凑的内部表示
    __slots__ = ["year", "month", "day"]

    def __init__(self, year, month, day):
        super(Date, self).__init__()
        self.year = year
        self.month = month
        self.day = day

#5 将名称封装到类中
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
        self.__private = 1 # 不会覆盖B.__private

    # 不会覆盖B.__private_method()
    def __private_method(self):
        pass    

#6 创建可管理的属性
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
            raise TypeError("expected a string")
        self._first_name = value

    # deleter function (optional)
    @first_name.deleter
    def first_name(self):
        raise AttributeError("cannot delete attribute")

a = Person("Mingle")
print(a.first_name) # calls the getter
a.first_name = "Jack" # calls the setter
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
            raise TypeError("expected a string")
        self._first_name = value

    # deleter function (optional)
    def del_first_name(self):
        raise AttributeError("cannot delete attribute")

a = Person("Mingle1")
print(a.get_first_name()) # calls the getter
a.set_first_name("Jack1") # calls the setter
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

#7 调用父类中的方法
class A:
    def __init__(self):
        self.x = 0

    def spam(self):
        print("A.spam")

class B(A):
    def __init__(self):
        super(B, self).__init__()
        self.y = 1

    def spam(self):
        print("B.spam")
        super().spam() # call parent spam()

class Proxy:
    def __init__(self, obj):
        super(Proxy, self).__init__()
        self._obj = obj

    # delegate attribute lookup to internal obj
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # delegate attribute assignment
    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value) # call original __setattr__
        else:
            setattr(self._obj, name, value)

class Base:
    def __init__(self):
        print("Base.__init__")

class A(Base):
    def __init__(self):
        Base.__init__(self)
        print("A.__init__")

class B(Base):
    def __init__(self):
        Base.__init__(self)
        print("B.__init__")

class C(A, B):
    def __init__(self):
        A.__init__(self)
        B.__init__(self)
        print("C.__init__")

c = C() # B.__init__()调用2次

class Base:
    def __init__(self):
        print("Base.__init__")

class A(Base):
    def __init__(self):
        super(A, self).__init__()
        print("A.__init__")

class B(Base):
    def __init__(self):
        super(B, self).__init__()
        print("B.__init__")

class C(A, B):
    def __init__(self):
        super(C, self).__init__()
        print("C.__init__")

c = C()
# 方法解析顺序(MRO)
# 1. 先检查子类再检查父类
# 2. 有多个父类时,按照MRO列表的顺序依次检查
# 3. 如果下一个待选的类出现了两个合法的选择,那么就从第一个父类中选取
print(C.__mro__)

#8 在子类中扩展属性
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
            raise TypeError("excepted a string")
        self._name = value

    @name.deleter
    def name(self):
        raise AttributeError("cannot delete attribute")

class SubPerson(Person):
    @property
    def name(self):
        print("getting name")
        return super().name

    @name.setter
    def name(self, value):
        print("setting name to", value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print("deleting name")
        super(SubPerson, SubPerson).name.__delete__(self)

s = SubPerson("Mingle")
print(s.name)
s.name = "Page"
# s.name = 42 # error

# 只想扩展属性中的一个方法
class SubPerson(Person):
    @Person.name.setter
    def name(self, value):
        print("setting name to", value)
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
            raise TypeError("excepted a string")
        instance.__dict__[self.name] = value

class Person:
    name = String("name")

    def __init__(self, name):
        super(Person, self).__init__()
        self.name = name

# extending a descriptor with a property
class SubPerson(Person):
    @property
    def name(self):
        print("getting name")
        return super().name

    @name.setter
    def name(self, value):
        print("setting name to", value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print("deleting name")
        super(SubPerson, SubPerson).name.__delete__(self)

#9 创建一种新形式的类属性或实例属性
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
            raise TypeError("expected an int")
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

class Point:
    x = Integer("x")
    y = Integer("y")

    def __init__(self, x, y):
        super(Point, self).__init__()
        self.x = x
        self.y = y

p = Point(2, 4)
print(p.x) # calls Point.x.__get__(p, Point)
print(p.y) # calls Point.y.__get__(p, Point)
p.y = 5 # calls Point.y.__set__(p, 5)
p.x = 2 # calls Point.x.__set__(p, 2)
print(p.x)
print(p.y)

#descriptor for a type-checked attribute
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
            raise TypeError("expected " + str(self.expected_type))
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

s = Stock("GOOG", 100, 789.65)
print(s.name)
print(s.shares)
print(s.price)

#10 让属性具有惰性求值的能力
