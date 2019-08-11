"""决策树

"""
import graphviz
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def entropy(data_df):
    x_value_list = set([data_df[i] for i in range(data_df.shape[0])])
    entropy_result = 0.0
    for x_value in x_value_list:
        p = float(data_df[data_df == x_value].shape[0]) / data_df.shape[0]
        logp = np.log2(p)
        entropy_result -= p * logp

    return entropy_result


def entropy_prob(prob):
    entropy_result = 0.0
    for value in prob:
        logp = np.log2(value)
        entropy_result -= value * logp

    return entropy_result


def iris_predict():
    """鸢尾花预测
    :return:
    """
    # 准备数据集
    iris = load_iris()
    # 获取特征集和分类标识
    features = iris.data
    labels = iris.target
    # 随机抽取 33% 的数据作为测试集，其余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=0)
    # 创建 CART 分类树
    clf = tree.DecisionTreeClassifier(criterion='gini')
    # 拟合构造 CART 分类树
    clf = clf.fit(train_features, train_labels)
    # 用 CART 分类树做预测
    test_predict = clf.predict(test_features)
    # 预测结果与测试集结果作比对
    score = metrics.accuracy_score(test_labels, test_predict)
    print("CART 分类树准确率 %.4lf" % score)


def boston_housing_price_predict():
    """波士顿房价预测
    :return:
    """
    # 准备数据集
    boston = load_boston()
    # 探索数据
    print(boston.feature_names)
    # 获取特征集和房价
    features = boston.data
    prices = boston.target
    # 随机抽取 33% 的数据作为测试集，其余为训练集
    train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)
    # 创建 CART 回归树
    dtr = tree.DecisionTreeRegressor()
    # 拟合构造 CART 回归树
    dtr.fit(train_features, train_price)
    # 预测测试集中的房价
    predict_price = dtr.predict(test_features)
    # 测试集的结果评价
    print('回归树二乘偏差均值:', metrics.mean_squared_error(test_price, predict_price))
    print('回归树绝对值偏差均值:', metrics.mean_absolute_error(test_price, predict_price))
    # 画决策树
    dot_data = tree.export_graphviz(dtr, out_file=None)
    graph = graphviz.Source(dot_data)
    # 输出分类树图示
    graph.view('Boston')


def titanic_analysis():
    """泰坦尼克生存预测
    :return:
    """
    # 数据加载
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # 数据探索
    print(train_data.info())
    print('-' * 30)
    print(train_data.describe())
    print('-' * 30)
    print(train_data.describe(include=['O']))
    print('-' * 30)
    print(train_data.head())
    print('-' * 30)
    print(train_data.tail())
    # 使用平均年龄来填充年龄中的 nan 值
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
    # 使用票价的均值填充票价中的 nan 值
    train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
    print(train_data['Embarked'].value_counts())
    # 使用登录最多的港口来填充登录港口的 nan 值
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)
    # 特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]
    # 用它将可以处理符号化的对象，将符号转成数字 0/1 进行表示
    dvec = DictVectorizer(sparse=False)
    train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
    print(dvec.feature_names_)
    # 构造 ID3 决策树
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # 决策树训练
    clf.fit(train_features, train_labels)
    # 模型预测&评估
    test_features = dvec.transform(test_features.to_dict(orient='record'))
    # 决策树预测
    pred_labels = clf.predict(test_features)
    # 得到决策树准确率
    acc_decision_tree = round(clf.score(train_features, train_labels), 6)
    print(u'score 准确率为 %.4lf' % acc_decision_tree)
    # 使用 K折交叉验证 统计决策树准确率
    print(u'cross_val_score K折交叉验证准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))


if __name__ == '__main__':
    prob = np.array([1 / 6, 5 / 6])
    print(entropy_prob(prob))
    data = np.array([1, 2, 2, 2, 2, 2])
    print(entropy(data))
    # iris_predict()
    # boston_housing_price_predict()
    titanic_analysis()
