"""Data Encoding and Processing.

"""

#1 读写CSV数据
import csv
import re
with open("data/stocks.csv") as f:
    f_csv = csv.reader(f)
    headers = [re.sub("[^a-zA-Z_]", "_", h) for h in next(f_csv)]
    print(headers)
    for row in f_csv:
        print(row)

# 命名元组
from collections import namedtuple
with open("data/stocks.csv") as f:
    f_csv = csv.reader(f)
    headers = [re.sub("[^a-zA-Z_]", "_", h) for h in next(f_csv)]
    row_tuple = namedtuple("Row", headers)
    for r in f_csv:
        row = row_tuple(*r)
        print(row)

# 字典序列
with open("data/stocks.csv") as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        print(row)

headers = ["Symbol", "Price", "Date", "Time", "Change", "Volume"];
rows = [("AA", "39.48", "6/11/2007", "9:36am", "-0.18", "181800"),
        ("AIG", "71.38", "6/11/2007", "9:36am", "-0.15", "195500"),
        ("AXP", "62.58", "6/11/2007", "9:36am", "-0.46", "935000"),
        ("BA", "98.31", "6/11/2007", "9:36am", "+0.12", "104800"),
        ("C", "53.08", "6/11/2007", "9:36am", "-0.25", "360900"),
        ("CAT", "78.29", "6/11/2007", "9:36am", "-0.23", "225400")
    ]
with open("data/stocks_write.csv", "w") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)

headers = ["Symbol", "Price", "Date", "Time", "Change", "Volume"];
rows = [{"Symbol": "AA", "Price": "39.48", "Date": "6/11/2007", "Time": "9:36am", "Change": "-0.18", "Volume": "181800"},
        {"Symbol": "AIG", "Price": "71.38", "Date": "6/11/2007", "Time": "9:36am", "Change": "-0.15", "Volume": "195500"},
        {"Symbol": "AXP", "Price": "62.58", "Date": "6/11/2007", "Time": "9:36am", "Change": "-0.46", "Volume": "935000"},
        {"Symbol": "BA", "Price": "98.31", "Date": "6/11/2007", "Time": "9:36am", "Change": "+0.12", "Volume": "104800"},
        {"Symbol": "C", "Price": "53.08", "Date": "6/11/2007", "Time": "9:36am", "Change": "-0.25", "Volume": "360900"},
        {"Symbol": "CAT", "Price": "78.29", "Date": "6/11/2007", "Time": "9:36am", "Change": "-0.23", "Volume": "225400"}
    ]
with open("data/stocks_write.csv", "w") as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

# 类型转换
col_types = [str, float, str, str, float, int]
with open("data/stocks.csv") as f:
    f_csv = csv.reader(f)
    headers = [re.sub("[^a-zA-Z_]", "_", h) for h in next(f_csv)]
    for row in f_csv:
        row = tuple(convert(value) for convert, value in zip(col_types, row))
        print(row)

filed_types = [("Price", float), ("Change", float), ("Volume", int)]
with open("data/stocks.csv") as f:
    for row in csv.DictReader(f):
        row.update((key, conversion(row[key])) for key, conversion in filed_types)
        print(row)

#2 读取JSON数据
import json
data = {
    "name": "ACME",
    "shares": 100,
    "price": 542.23
}
json_str = json.dumps(data)
print(json_str)
data = json.loads(json_str)
print(data)

with open("data/data.json", "w") as f:
    json.dump(data, f)
with open("data/data.json", "r") as f:
    data = json.load(f)
    print(data)

print(json.dumps(False))
d = {
    "a": True,
    "b": "Hello",
    "c": None
}
print(json.dumps(d))

from urllib import request
headers = {
    "apikey": "dbd04f5efcd6d4e698678170e5b149eb"
}
req = request.Request("http://apis.baidu.com/tianyiweather/basicforecast/weatherapi?area=101010100", headers=headers);
u = request.urlopen(req)
resp = json.loads(u.read().decode("utf-8"))
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
    "name": "ACME",
    "shares": 100,
    "price": 542.23
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
    d = {"__classname__": type(obj).__name__}
    d.update(vars(obj))
    return d

# 取回实例
classes = {
    "Point": Point
}

def unserialize_object(d):
    clsname = d.pop("__classname__", None)
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
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

# u = request.urlopen("http://planet.python.org/rss20.xml")
# u = open("data/big.xml", "rt", encoding="utf-8")
# doc = parse(u)
doc = parse("data/big.xml")

for item in doc.iterfind("channel/item"):
    title = item.findtext("title")
    date = item.findtext("pubDate")
    link = item.findtext("link")
    print(title)
    print(date)
    print(link)
    print()
e = doc.find("channel/title")
print(e)
print(e.tag)
print(e.text)
print(e.get("some_attribute"))

#4 以增量方式解析大型XML文件
from xml.etree.ElementTree import iterparse

def parse_and_remove(filename, path):
    path_parts = path.split("/")
    doc = iterparse(filename, ("start", "end"))
    # skip the root element
    next(doc)

    tag_stack = []
    elem_stack = []
    for event, elem in doc:
        if "start" == event:
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif "end" == event:
            if tag_stack == path_parts:
                yield elem
                elem_stack[-2].remove(elem)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass

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
    "name": "GOOG",
    "shares": 100,
    "price": 780.6
}
e = dict_to_xml("stock", s)
print(e)

from xml.etree.ElementTree import tostring
print(tostring(e))

def dict_to_xml_str(tag, d):
    parts = ["<{}>".format(tag)]
    for key, val in d.items():
        parts.append("<{0}>{1}</{0}>".format(key, val))
    parts.append("</{}>".format(tag))
    return "".join(parts)
d = {"name": "<spam>"}
e = dict_to_xml_str("item", d)
print(e)
e = dict_to_xml("item", d)
print(tostring(e))

# 转义字符处理
from xml.sax.saxutils import escape, unescape
print(escape("<spam>"))
print(unescape("<spam>"))

#6 解析,修改和重写XML
doc = parse("data/write.xml")
root = doc.getroot()
print(root)
root.remove(root.find("sri"))
root.remove(root.find("cr"))
print(root.getchildren().index(root.find("nm")))
e = Element("spam")
e.text = "this is a test"
root.insert(2, e)
doc.write("data/write_new.xml", xml_declaration=True)

#7 用命名空间来解析XML文档
doc = parse("data/namespace.xml")
print(doc.findtext("author"))
print(doc.find("content"))
print(doc.find("content/html"))
print(doc.find("content/{http://www.w3.org/1999/xhtml}html"))
print(doc.findtext("content/{http://www.w3.org/1999/xhtml}html/head/title"))
print(doc.findtext("content/{http://www.w3.org/1999/xhtml}html/" 
    "{http://www.w3.org/1999/xhtml}head/{http://www.w3.org/1999/xhtml}title"))

class XMLNamespace:
    def __init__(self, **kwargs):
        self.namespaces = {}
        for name, uri in kwargs.items():
            self.register(name, uri)
    
    def register(self, name, uri):
        self.namespaces[name] = "{" + uri + "}"

    def __call__(self, path):
        return path.format_map(self.namespaces)
ns = XMLNamespace(html="http://www.w3.org/1999/xhtml")
doc.find(ns("content/{html}html"))
doc.findtext(ns("content/{html}html/{html}head/{html}title"))

for event, element in iterparse("data/namespace.xml", ("end", "start-ns", "end-ns")):
    print(event, element)

#8 同关系型数据库进行交互
stocks = [
    ("GOOG", 100, 500.1),
    ("AAPL", 50, 545.75),
    ("FB", 150, 7.75),
    ("HPQ", 75, 33.2)
]
import sqlite3
db = sqlite3.connect("database.db")
c = db.cursor()
c.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'")
tablenames = [name[0] for name in c.fetchall()]
create_table = "portfoio"
if create_table not in tablenames:
    c.execute("create table {} (symbol text, shares integer, price real)".format(create_table))
c.execute("SELECT COUNT(0) FROM {}".format(create_table))
table_count = c.fetchone()[0]
if table_count <= 0:
    c.executemany("insert into portfoio values (?,?,?)", stocks)
for row in db.execute("select * from {}".format(create_table)):
    print(row)
min_price = 100
for row in db.execute("select * from portfoio where price >= ?", (min_price,)):
    print(row)

#9 编码和解码十六进制数字
