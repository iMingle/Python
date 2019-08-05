"""数据清洗

"""
import pandas as pd


df = pd.read_excel('./data/account.xlsx')
print('待清洗数据')
print(df)

df.rename(columns={0: 'no', 1: 'name', 2: 'age', 3: 'weight', 4: 'm0006', 5: 'm0612',
                   6: 'm1218', 7: 'f0006', 8: 'f0612', 9: 'f1218'}, inplace=True)
df.drop(['no'], axis=1, inplace=True)
df.drop(['\t'], axis=1, inplace=True)
print("列重命名,删除序号列")
print(df)
# 删除全空的行
df.dropna(how='all', inplace=True)
print('删除全空的行')
print(df)
# 平均年龄填充
df['age'].fillna(int(df['age'].mean()), inplace=True)
# 用最高频的数据填充
age_maxf = df['age'].value_counts().index[0]
df['age'].fillna(age_maxf, inplace=True)
print('年龄空值处理')
print(df)

# 获取 weight 数据列中单位为 lbs 的数据
rows_with_lbs = df['weight'].str.contains('lbs').fillna(False)
# 将 lbs 转换为 kgs, 2.2lbs=1kgs
for i, lbs_row in df[rows_with_lbs].iterrows():
    # 截取从头开始到倒数第三个字符之前，即去掉lbs
    weight = int(float(lbs_row['weight'][:-3]) / 2.2)
    df.at[i, 'weight'] = '{}kgs'.format(weight)
print('weight列单位统一')
print(df)

# 切分名字，删除源数据列
df[['first_name', 'last_name']] = df['name'].str.split(expand=True)
df.drop('name', axis=1, inplace=True)
last_name = df.pop('last_name')
df.insert(0, 'last_name', last_name)
first_name = df.pop('first_name')
df.insert(0, 'first_name', first_name)
# 删除非 ASCII 字符
df['first_name'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
df['last_name'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
# 删除重复数据行
df.drop_duplicates(['first_name', 'last_name'], inplace=True)
print(df)
