"""Data Encoding and Processing.

"""

#1 读写CSV数据
import csv
import re
with open('data/stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = [re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv)]
    print(headers)
    for row in f_csv:
        print(row)

# 命名元组
from collections import namedtuple
with open('data/stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = [re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv)]
    row_tuple = namedtuple('Row', headers)
    for r in f_csv:
        row = row_tuple(*r)
        print(row)

# 字典序列
with open('data/stocks.csv') as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        print(row)

headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume'];
rows = [('AA', '39.48', '6/11/2007', '9:36am', '-0.18', '181800'),
        ('AIG', '71.38', '6/11/2007', '9:36am', '-0.15', '195500'),
        ('AXP', '62.58', '6/11/2007', '9:36am', '-0.46', '935000'),
        ('BA', '98.31', '6/11/2007', '9:36am', '+0.12', '104800'),
        ('C', '53.08', '6/11/2007', '9:36am', '-0.25', '360900'),
        ('CAT', '78.29', '6/11/2007', '9:36am', '-0.23', '225400')
    ]
with open('data/stocks_write.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)

headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume'];
rows = [{'Symbol': 'AA', 'Price': '39.48', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '-0.18', 'Volume': '181800'},
        {'Symbol': 'AIG', 'Price': '71.38', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '-0.15', 'Volume': '195500'},
        {'Symbol': 'AXP', 'Price': '62.58', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '-0.46', 'Volume': '935000'},
        {'Symbol': 'BA', 'Price': '98.31', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '+0.12', 'Volume': '104800'},
        {'Symbol': 'C', 'Price': '53.08', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '-0.25', 'Volume': '360900'},
        {'Symbol': 'CAT', 'Price': '78.29', 'Date': '6/11/2007', 'Time': '9:36am', 'Change': '-0.23', 'Volume': '225400'}
    ]
with open('data/stocks_write.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

# 类型转换
col_types = [str, float, str, str, float, int]
with open('data/stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = [re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv)]
    for row in f_csv:
        row = tuple(convert(value) for convert, value in zip(col_types, row))
        print(row)

filed_types = [('Price', float), ('Change', float), ('Volume', int)]
with open('data/stocks.csv') as f:
    for row in csv.DictReader(f):
        row.update((key, conversion(row[key])) for key, conversion in filed_types)
        print(row)

#2 读取JSON数据
import json
data = {
    'name': 'ACME',
    'shares': 100,
    'price': 542.23
}
json_str = json.dumps(data)
print(json_str)
data = json.loads(json_str)
print(data)

with open('data/data.json', 'w') as f:
    json.dump(data, f)
with open('data/data.json', 'r') as f:
    data = json.load(f)
    print(data)

print(json.dumps(False))
d = {
    'a': True,
    'b': 'Hello',
    'c': None
}
print(json.dumps(d))

from urllib import request
headers = {
    'apikey': 'dbd04f5efcd6d4e698678170e5b149eb'
}
req = request.Request('http://apis.baidu.com/tianyiweather/basicforecast/weatherapi?area=101010100', headers=headers);
u = request.urlopen(req)
resp = json.loads(u.read().decode('utf-8'))
from pprint import pprint
pprint(resp)

s = '{"name": "ACME", "shares": 100, "price": 542.23}'
from collections import OrderedDict
data = json.loads(s, object_pairs_hook=OrderedDict)
print(data)

class JSONObject:
    """json object"""
    def __init__(self, d):
        super(JSONObject, self).__init__()
        self.__dict__ = d
data = json.loads(s, object_hook=JSONObject)
print(data.name)
print(data.shares)
print(data.price)

data = {
    'name': 'ACME',
    'shares': 100,
    'price': 542.23
}
print(json.dumps(data))
print(json.dumps(data, indent=4))
print(json.dumps(data, indent=4, sort_keys=True))

# 类实例一般是无法序列化为JSON的
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
p = Point(2, 3)
print(vars(p))
# json.dumps(p) # error

def serialize_instance(obj):
    d = {'__classname__': type(obj).__name__}
    d.update(vars(obj))
    return d

# 取回实例
classes = {
    'Point': Point
}

def unserialize_object(d):
    clsname = d.pop('__classname__', None)
    if clsname:
        cls = classes[clsname]
        obj = cls.__new__(cls) # make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d
p = Point(3, 4)
s = json.dumps(p, default=serialize_instance)
print(s)
a = json.loads(s, object_hook=unserialize_object)
print(a)
print(a.x)
print(a.y)

#3 解析简单的XML文档
from xml.etree.ElementTree import parse
import sys
from base import io


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# u = request.urlopen('http://planet.python.org/rss20.xml')
# u = open('data/big.xml', 'rt', encoding='utf-8')
# doc = parse(u)
doc = parse('data/big.xml')

for item in doc.iterfind('channel/item'):
    title = item.findtext('title')
    date = item.findtext('pubDate')
    link = item.findtext('link')
    print(title)
    print(date)
    print(link)
    print()
e = doc.find('channel/title')
print(e)
print(e.tag)
print(e.text)
print(e.get('some_attribute'))

#4 以增量方式解析大型XML文件
from xml.etree.ElementTree import iterparse

def parse_and_remove(filename, path):
    path_parts = path.split('/')
    doc = iterparse(filename, ('start', 'end'))
    # skip the root element
    next(doc)

    tag_stack = []
    elem_stack = []
    for event, elem in doc:
        if 'start' == event:
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif 'end' == event:
            if tag_stack == path_parts:
                yield elem
                elem_stack[-2].remove(elem)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass
from collections import Counter
potholes_by_zip = Counter()
doc = parse('data/potholes.xml')
for pothole in doc.iterfind('row/row'):
    potholes_by_zip[pothole.findtext('zip')] += 1
for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)

# 占用内存小的方式
data = parse_and_remove('data/potholes.xml', 'row/row')
potholes_by_zip = Counter()
for pothole in data:
    potholes_by_zip[pothole.findtext('zip')] += 1
for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)

#5 将字典转换为XML
from xml.etree.ElementTree import Element

def dict_to_xml(tag, d):
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return elem
s = {
    'name': 'GOOG',
    'shares': 100,
    'price': 780.6
}
e = dict_to_xml('stock', s)
print(e)

from xml.etree.ElementTree import tostring
print(tostring(e))

def dict_to_xml_str(tag, d):
    parts = ['<{}>'.format(tag)]
    for key, val in d.items():
        parts.append('<{0}>{1}</{0}>'.format(key, val))
    parts.append('</{}>'.format(tag))
    return ''.join(parts)
d = {'name': '<spam>'}
e = dict_to_xml_str('item', d)
print(e)
e = dict_to_xml('item', d)
print(tostring(e))

# 转义字符处理
from xml.sax.saxutils import escape, unescape
print(escape('<spam>'))
print(unescape('<spam>'))

#6 解析,修改和重写XML
doc = parse('data/write.xml')
root = doc.getroot()
print(root)
root.remove(root.find('sri'))
root.remove(root.find('cr'))
print(root.getchildren().index(root.find('nm')))
e = Element('spam')
e.text = 'this is a test'
root.insert(2, e)
doc.write('data/write_new.xml', xml_declaration=True)

#7 用命名空间来解析XML文档
doc = parse('data/namespace.xml')
print(doc.findtext('author'))
print(doc.find('content'))
print(doc.find('content/html'))
print(doc.find('content/{http://www.w3.org/1999/xhtml}html'))
print(doc.findtext('content/{http://www.w3.org/1999/xhtml}html/head/title'))
print(doc.findtext('content/{http://www.w3.org/1999/xhtml}html/' 
    '{http://www.w3.org/1999/xhtml}head/{http://www.w3.org/1999/xhtml}title'))

class XMLNamespace:
    def __init__(self, **kwargs):
        self.namespaces = {}
        for name, uri in kwargs.items():
            self.register(name, uri)
    
    def register(self, name, uri):
        self.namespaces[name] = '{' + uri + '}'

    def __call__(self, path):
        return path.format_map(self.namespaces)
ns = XMLNamespace(html='http://www.w3.org/1999/xhtml')
doc.find(ns('content/{html}html'))
doc.findtext(ns('content/{html}html/{html}head/{html}title'))

for event, element in iterparse('data/namespace.xml', ('end', 'start-ns', 'end-ns')):
    print(event, element)

#8 同关系型数据库进行交互
stocks = [
    ('GOOG', 100, 500.1),
    ('AAPL', 50, 545.75),
    ('FB', 150, 7.75),
    ('HPQ', 75, 33.2)
]
import sqlite3
db = sqlite3.connect('database.db')
c = db.cursor()
c.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'")
tablenames = [name[0] for name in c.fetchall()]
create_table = 'portfoio'
if create_table not in tablenames:
    c.execute('create table {} (symbol text, shares integer, price real)'.format(create_table))
c.execute('SELECT COUNT(0) FROM {}'.format(create_table))
table_count = c.fetchone()[0]
if table_count <= 0:
    c.executemany('insert into portfoio values (?,?,?)', stocks)
for row in db.execute('select * from {}'.format(create_table)):
    print(row)
min_price = 100
for row in db.execute('select * from portfoio where price >= ?', (min_price,)):
    print(row)

#9 编码和解码十六进制数字
s = b'hello'
import binascii
h = binascii.b2a_hex(s)
print(h)
print(binascii.a2b_hex(h))

import base64
h = base64.b16encode(s)
print(h)
print(base64.b16decode(h))

#10 Base64编码和解码
# Base64编码对二进制数据做编码解码操作
s = b'hello'
a = base64.b64encode(s)
print(a)
print(base64.b64decode(a))
a = base64.b64encode(s).decode('ascii')
print(a)

#11 读写二进制结构的数组
from struct import Struct

def write_records(records, format, f):
    """write a sequence of tuples to a binary file of structures."""
    record_struct = Struct(format)
    for r in records:
        f.write(record_struct.pack(*r))
if '__main__' == __name__:
    records = [
        (1, 2.3, 4.5),
        (6, 7.8, 9.0),
        (12, 13.4, 56.7)
    ]
    with open('data/data.b', 'wb') as f:
        write_records(records, '<idd', f)

def read_records(format, f):
    record_struct = Struct(format)
    chunks = iter(lambda: f.read(record_struct.size), b'')
    return (record_struct.unpack(chunk) for chunk in chunks)
if '__main__' == __name__:
    with open('data/data.b', 'rb') as f:
        for rec in read_records('<idd', f):
            print(rec)

# 将文件全部读取到一个字节串中
def unpack_records(format, data):
    record_struct = Struct(format)
    return (record_struct.unpack_from(data, offset) for offset in range(0, len(data), record_struct.size))
if '__main__' == __name__:
    with open('data/data.b', 'rb') as f:
        data = f.read()
        for rec in unpack_records('<idd', data):
            print(rec)

record_struct = Struct('<idd')
print(record_struct.size)
a = record_struct.pack(1, 2.0, 3.0)
print(a)
print(record_struct.unpack(a))

import struct
a = struct.pack('<idd', 1, 2.0, 3.0)
print(a)
print(struct.unpack('<idd', a))

f = open('data/data.b', 'rb')
chunks = iter(lambda: f.read(20), b'')
for chk in chunks:
    print(chk)

from collections import namedtuple

record = namedtuple('record', ['kind', 'x', 'y'])
with open('data/data.b', 'rb') as f:
    records = (record(*r) for r in read_records('<idd', f))
    for r in records:
        print(r.kind, r.x, r.y)

import numpy as np
f = open('data/data.b', 'rb')
records = np.fromfile(f, dtype='<i,<d,<d')
print(records)
print(records[0])
print(records[1])

#12 读取嵌套型和大小可变的二进制结构
import struct
import itertools

polys = [
        [ (1.0, 2.5), (3.5, 4.0), (2.5, 1.5) ],
        [ (7.0, 1.2), (5.1, 3.0), (0.5, 7.5), (0.8, 9.0) ],
        [ (3.4, 6.3), (1.2, 0.5), (4.6, 9.2) ],
    ]

def write_polys(filename, polys):
    """determine bounding box"""
    flattened = list(itertools.chain(*polys))
    min_x = min(x for x, y in flattened)
    max_x = max(x for x, y in flattened)
    min_y = min(y for x, y in flattened)
    max_y = max(y for x, y in flattened)

    with open(filename, 'wb') as f:
        f.write(struct.pack('<iddddi', 0x1234, min_x, min_y, max_x, max_y, len(polys)))
        for poly in polys:
            size = len(poly) * struct.calcsize('<dd')
            f.write(struct.pack('<i', size+4))
            for pt in poly:
                f.write(struct.pack('<dd', *pt))
write_polys('data/polys.bin', polys)

def read_polys(filename):
    with open('data/polys.bin', 'rb') as f:
        # read the header
        header = f.read(40)
        file_code, min_x, min_y, max_x, max_y, num_polys = struct.unpack('<iddddi', header)
        polys = []
        for n in range(num_polys):
            pbytes, = struct.unpack('<i', f.read(4))
            poly = []
            for m in range(pbytes // 16):
                pt = struct.unpack('<dd', f.read(16))
                poly.append(pt)
            polys.append(poly)
    return polys
polys = read_polys('data/polys.bin')
print(polys)

# 高级解决方案
class StructField:
    """descriptor representing a simple structure field"""
    def __init__(self, format, offset):
        super(StructField, self).__init__()
        self.format = format
        self.offset = offset

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            r = struct.unpack_from(self.format, instance._buffer, self.offset)
            return r[0] if len(r) == 1 else r

class Structure():
    def __init__(self, bytedata):
        super(Structure, self).__init__()
        self._buffer = memoryview(bytedata)

if __name__ == '__main__':
    class PolyHeader(Structure):
        file_code = StructField('<i', 0)
        min_x = StructField('<d', 4)
        min_y = StructField('<d', 12)
        max_x = StructField('<d', 20)
        max_y = StructField('<d', 28)
        num_polys = StructField('<i', 36)

    f = open('data/polys.bin', 'rb')
    data = f.read()
    
    phead = PolyHeader(data)
    print(phead.file_code == 0x1234)
    print('min_x=', phead.min_x)
    print('max_x=', phead.max_x)
    print('min_y=', phead.min_y)
    print('max_y=', phead.max_y)
    print('num_polys=', phead.num_polys)

class StructureMeta(type):
    """Metaclass that automatically creates StructField descriptos."""
    def __init__(self, clsname, bases, clsdict):
        fields = getattr(self, '_fields_', [])
        byte_order = ''
        offset = 0
        for format, fieldname in fields:
            if format.startswith(('<', '>', '!', '@')):
                byte_order = format[0]
                format = format[1:]
            format = byte_order + format
            setattr(self, fieldname, StructField(format, offset))
            offset += struct.calcsize(format)
        setattr(self, 'struct_size', offset)

class Structure(metaclass=StructureMeta):
    def __init__(self, bytedata):
        super(Structure, self).__init__()
        self._buffer = bytedata

    @classmethod
    def from_file(cls, f):
        return cls(f.read(cls.struct_size))
if __name__ == '__main__':
    class PolyHeader(Structure):
        _fields_ = [
            ('<i', 'file_code'),
            ('d', 'min_x'),
            ('d', 'min_y'),
            ('d', 'max_x'),
            ('d', 'max_y'),
            ('i', 'num_polys')
        ]

    f = open('data/polys.bin', 'rb')
    phead = PolyHeader.from_file(f)
    print(phead.file_code == 0x1234)
    print('min_x=', phead.min_x)
    print('max_x=', phead.max_x)
    print('min_y=', phead.min_y)
    print('max_y=', phead.max_y)
    print('num_polys=', phead.num_polys)

class NestedStruct:
    """descriptor representing a nested structure"""
    def __init__(self, name, struct_type, offset):
        self.name = name
        self.struct_type = struct_type
        self.offset = offset

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            data = instance._buffer[self.offset:self.offset+self.struct_type.struct_size]
            result = self.struct_type(data)
            setattr(instance, self.name, result)
            return result

class StructureMeta(type):
    'metaclass that automatically creates StructField descriptors'
    def __init__(self, clsname, bases, clsdict):
        fields = getattr(self, '_fields_', [])
        byte_order = ''
        offset = 0
        for format, fieldname in fields:
            if isinstance(format, StructureMeta):
                setattr(self, fieldname, NestedStruct(fieldname, format, offset))
                offset += format.struct_size
            else:
                if format.startswith(('<', '>', '!', '@')):
                    byte_order = format[0]
                    format = format[1:]
                format = byte_order + format
                setattr(self, fieldname, StructField(format, offset))
                offset += struct.calcsize(format)
        setattr(self, 'struct_size', offset)

class Structure(metaclass=StructureMeta):
    def __init__(self, bytedata):
        self._buffer = memoryview(bytedata)

    @classmethod
    def from_file(cls, f):
        return cls(f.read(cls.struct_size))

if __name__ == '__main__':
    class Point(Structure):
        _fields_ = [
            ('<d', 'x'),
            ('d', 'y')
        ]

    class PolyHeader(Structure):
        _fields_ = [
            ('<i', 'file_code'),
            (Point, 'min'),
            (Point, 'max'),
            ('i', 'num_polys')
        ]

    f = open('data/polys.bin', 'rb')
    phead = PolyHeader.from_file(f)
    print(phead.file_code == 0x1234)
    print('min.x=', phead.min.x)
    print('max.x=', phead.max.x)
    print('min.y=', phead.min.y)
    print('max.y=', phead.max.y)
    print('num_polys=', phead.num_polys)

class SizedRecord:
    def __init__(self, bytedata):
        self._buffer = memoryview(bytedata)

    @classmethod
    def from_file(cls, f, size_fmt, includes_size=True):
        sz_nbytes = struct.calcsize(size_fmt)
        sz_bytes = f.read(sz_nbytes)
        sz, = struct.unpack(size_fmt, sz_bytes)
        buf = f.read(sz - includes_size * sz_nbytes)
        return cls(buf)

    def iter_as(self, code):
        if isinstance(code, str):
            s = struct.Struct(code)
            for off in range(0, len(self._buffer), s.size):
                yield s.unpack_from(self._buffer, off)
        elif isinstance(code, StructureMeta):
            size = code.struct_size
            for off in range(0, len(self._buffer), size):
                data = self._buffer[off:off+size]
                yield code(data)
if __name__ == '__main__':
    class Point(Structure):
        _fields_ = [
            ('<d', 'x'),
            ('d', 'y')
        ]

    class PolyHeader(Structure):
        _fields_ = [
            ('<i', 'file_code'),
            (Point, 'min'),
            (Point, 'max'),
            ('i', 'num_polys')
        ]

    def read_polys(filename):
        polys = []
        with open(filename, 'rb') as f:
            phead = PolyHeader.from_file(f)
            for n in range(phead.num_polys):
                rec = SizedRecord.from_file(f, '<i')
                poly = [ (p.x, p.y) for p in rec.iter_as(Point) ]
                polys.append(poly)
        return polys

    polys = read_polys('data/polys.bin')
    print(polys)

#13 数据汇总和统计
import pandas
rats = pandas.read_csv('data/rats.csv', skip_footer=1)
print(rats)
# investigate range of values for a certain field
print(rats['Current Activity'].unique())
# filter the data
crew_dispatched = rats[rats['Current Activity'] == 'Dispatch Crew']
print(len(crew_dispatched))
# find 10 most rat-infested ZIP codes in Chicago
# print(crew_dispatched['ZIP Code'].value_counts()[0:10])
# group by completion date
dates = crew_dispatched.groupby('Completion Date')
print(len(dates))
# determine counts on each day
date_counts = dates.size()
print(date_counts[0:10])
# sort the counts
date_counts.sort()
print(date_counts[-10:0])
