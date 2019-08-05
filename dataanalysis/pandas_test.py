"""Pandas

"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandasql import sqldf, load_meat, load_births


x1 = Series([1, 2, 3, 4])
x2 = Series(data=[1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
x3 = Series(d)
print('Series数据结构')
print(x1)
print(x2)
print(x3)

data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [30, 98, 96, 77, 90]}
df1 = DataFrame(data)
df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'],
                columns=['English', 'Math', 'Chinese'])
print('DataFrame数据结构')
print(df1)

# 数据清洗
df2 = df2.drop(columns=['Chinese'])
df2 = df2.drop(index=['ZhangFei'])
df2.rename(columns={'Chinese': 'YuWen', 'English': 'Yingyu'}, inplace=True)
df2 = df2.drop_duplicates()
# 修改数据格式
# df2['Chinese'].astype('str')
# df2['Chinese'].astype(np.int64)
# 删除左右两边空格
# df2['Chinese'] = df2['Chinese'].map(str.strip)
# 删除左边空格
# df2['Chinese'] = df2['Chinese'].map(str.lstrip)
# 删除右边空格
# df2['Chinese'] = df2['Chinese'].map(str.rstrip)
# df2['Chinese'] = df2['Chinese'].str.strip('$')
# 全部大写
df2.columns = df2.columns.str.upper()
# 全部小写
df2.columns = df2.columns.str.lower()
# 首字母大写
df2.columns = df2.columns.str.title()
# df2['Chinese'] = df2['Chinese'].apply(str.upper)

print(df2)
print(df2.isnull().any())

# 数据统计
df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
print(df1)
print(df1.describe())

# 数据表合并
df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
df2 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2': range(5)})
df3 = pd.merge(df1, df2, on='name')
print(df1)
print(df2)
print('按name合并')
print(df3)
df3 = pd.merge(df1, df2, how='left')
print('左连接')
print(df3)
df3 = pd.merge(df1, df2, how='right')
print('右连接')
print(df3)
df3 = pd.merge(df1, df2, how='outer')
print('外连接')
print(df3)

df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
pysqldf = lambda sql: sqldf(sql, globals())
sql = "SELECT * FROM df1 WHERE name ='ZhangFei'"
print(pysqldf(sql))
