"""Test

"""

# Lambda表达式,用来创建新的函数对象,只能使用表达式
def make_repeater(n):
    return lambda s: s*n

twice = make_repeater(2)
print(twice("work"))
print(twice(5))

points = [{"x": 2, "y": 3}, {"x": 4, "y": 1}, {"x": 0, "y": 1}, {"x": 1, "y": 1}]
points.sort(key = lambda point : point["x"]);
print(points)

# 列表综合
listone = [2, 3, 4]
listtwo = [2*i for i in listone if i > 2]
print(listtwo)

def powersum(power, *args):
    '''Return the sum of each argument raised to specified power.'''
    total = 0
    for i in args:
        total += pow(i, power)
    return total

print(powersum(2, 3, 4))
print(powersum(2, 10))

# exec语句用来执行储存在字符串或文件中的Python语句
exec("print('hello')")
# eval函数用来执行存储在字符串中的Python表达式
print(eval('2*3'))

# assert
list = ["item"]
assert len(list) >= 1
list.pop()
# assert len(list) >= 1

# repr函数用来取得对象的规范字符串表示,注意,在大多数时候有eval(repr(object)) == object.
i = []
i.append("item")
print(i)
print(repr(i))
print(eval(repr(i)))
print(eval(repr(i)) == i)

from operator import itemgetter

from itertools import groupby
from datetime import datetime, timedelta

groupMapStore = [
    {"id": 1, "name": "name1", "s_id": 100},
    {"id": 1, "name": "name1", "s_id": 101},
    {"id": 1, "name": "name1", "s_id": 102},
    {"id": 1, "name": "name1", "s_id": 103},
    {"id": 2, "name": "name2", "s_id": 104},
    {"id": 2, "name": "name2", "s_id": 105},
    {"id": 3, "name": "name3", "s_id": 106}
]

orders = [
    {"company_id": 100, "date": "2016-06-01", "price1": 12.3, "price2": 23.4},
    {"company_id": 100, "date": "2016-06-01", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-02", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-03", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-04", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-05", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-06", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-07", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-08", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-09", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-10", "price1": 10.3, "price2": 22.4},
    {"company_id": 100, "date": "2016-06-11", "price1": 10.3, "price2": 22.4},

    {"company_id": 101, "date": "2016-06-01", "price1": 12.3, "price2": 23.4},
    {"company_id": 101, "date": "2016-06-02", "price1": 12.3, "price2": 23.4},
    {"company_id": 102, "date": "2016-06-08", "price1": 12.3, "price2": 23.4},
    {"company_id": 103, "date": "2016-06-09", "price1": 12.3, "price2": 23.4},
    {"company_id": 104, "date": "2016-06-10", "price1": 12.3, "price2": 23.4},
    {"company_id": 105, "date": "2016-06-11", "price1": 12.3, "price2": 23.4},
    {"company_id": 106, "date": "2016-06-12", "price1": 12.3, "price2": 23.4}
]

groupMap = {}
for id, items in groupby(groupMapStore, key=itemgetter("id")):
    arr = []
    name = ""
    print(items)
    for item in items:
        item.pop("id")
        name = item.pop("name")
        sid = item.pop("s_id")
        arr.append(sid)
    groupMap[str(id) + "_" + name] = arr;

for order in orders:
    order["date"] = datetime.strptime(order["date"], "%Y-%m-%d")
print(groupMap)
print(orders)

date = "2016-07-01"
d = datetime.strptime(date, "%Y-%m-%d")
print(d + timedelta(days=1))

def week_of_month(year, month, day):
    end = int(datetime(year, month, day).strftime("%W"))
    start = int(datetime(year, month, 1).strftime("%W"))
    return end - start + 1
print(week_of_month(2016, 8, 1))
