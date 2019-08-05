"""数据展示

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def jointplot():
    # 散点图数据准备
    N = 1000
    x = np.random.randn(N)
    y = np.random.randn(N)
    # 用 Matplotlib 画散点图
    plt.scatter(x, y, marker='o')
    plt.show()
    # 用 Seaborn 画散点图
    df = pd.DataFrame({'x': x, 'y': y})
    sns.jointplot(x='x', y='y', data=df, kind='scatter')
    plt.show()


def lineplot():
    # 折线图数据准备
    x = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    y = [5, 3, 6, 20, 17, 16, 19, 30, 32, 35]
    # 使用 Matplotlib 画折线图
    plt.plot(x, y)
    plt.show()
    # 使用 Seaborn 画折线图
    df = pd.DataFrame({'x': x, 'y': y})
    sns.lineplot(x="x", y="y", data=df)
    plt.show()


def distplot():
    # 直方图数据准备
    a = np.random.randn(100)
    s = pd.Series(a)
    # 用 Matplotlib 画直方图
    plt.hist(s)
    plt.show()
    # 用 Seaborn 画直方图
    sns.distplot(s, kde=False)
    plt.show()
    sns.distplot(s, kde=True)
    plt.show()


def barplot():
    # 条形图数据准备
    x = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
    y = [5, 4, 8, 12, 7]
    # 用 Matplotlib 画条形图
    plt.bar(x, y)
    plt.show()
    # 用 Seaborn 画条形图
    sns.barplot(x, y)
    plt.show()


def boxplot():
    # 箱线图数据准备
    # 生成 0-1 之间的 10*4 维度数据
    data = np.random.normal(size=(10, 4))
    lables = ['A', 'B', 'C', 'D']
    # 用 Matplotlib 画箱线图
    plt.boxplot(data, labels=lables)
    plt.show()
    # 用 Seaborn 画箱线图
    df = pd.DataFrame(data, columns=lables)
    sns.boxplot(data=df)
    plt.show()


def pie():
    # 饼图数据准备
    nums = [25, 37, 33, 37, 6]
    labels = ['High-school', 'Bachelor', 'Master', 'Ph.d', 'Others']
    # 用 Matplotlib 画饼图
    plt.pie(x=nums, labels=labels)
    plt.show()


def heatmap():
    """该数据集记录了 1949 年到 1960 年期间，每个月的航班乘客的数量。
    :return:
    """
    # 热力图数据准备
    flights = sns.load_dataset("flights")
    data = flights.pivot('year', 'month', 'passengers')
    # 用 Seaborn 画热力图
    sns.heatmap(data)
    plt.show()


def plot():
    # 蜘蛛图数据准备
    labels = np.array([u" 推进 ", "KDA", u" 生存 ", u" 团战 ", u" 发育 ", u" 输出 "])
    stats = [83, 61, 95, 67, 76, 88]
    # 画图数据准备，角度、状态值
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    # 用 Matplotlib 画蜘蛛图
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # o: 圆点,--: 虚线
    ax.plot(angles, stats, 'o--', linewidth=2)
    # alpha: 透明度
    ax.fill(angles, stats, alpha=0.25)
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    plt.show()


def bivariate():
    """二元变量
    这个数据集记录了不同顾客在餐厅的消费账单及小费情况。代码中 total_bill 保存了客户的账单金额，tip 是该客户给出的小费金额。
    我们可以用 Seaborn 中的 jointplot 来探索这两个变量之间的关系
    :return:
    """
    # 二元变量分布图数据准备
    tips = sns.load_dataset("tips")
    print(tips.head(10))
    # 用 Seaborn 画二元变量分布图（散点图，核密度图，Hexbin 图）
    sns.jointplot(x="total_bill", y="tip", data=tips, kind='scatter')
    sns.jointplot(x="total_bill", y="tip", data=tips, kind='kde')
    sns.jointplot(x="total_bill", y="tip", data=tips, kind='hex')
    plt.show()


def pairplot():
    """鸢尾花数据集。鸢尾花可以分成 Setosa、Versicolour 和 Virginica 三个品种，在这个数据集中，针对每一个品种，都有 50 个数据，
    每个数据中包括了 4 个属性，分别是花萼长度、花萼宽度、花瓣长度和花瓣宽度。通过这些数据，需要你来预测鸢尾花卉属于三个品种中的哪一种
    :return:
    """
    # 成对关系数据准备
    iris = sns.load_dataset('iris')
    print(iris)
    print(iris[iris['species'] == 'virginica'].index.size)
    # 用 Seaborn 画成对关系
    sns.pairplot(iris)
    plt.show()


if __name__ == '__main__':
    plot()
