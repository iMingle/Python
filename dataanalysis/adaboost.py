"""AdaBoost
AdaBoost 的英文全称是 Adaptive Boosting，中文含义是自适应提升算法

集成的含义就是集思广益，博取众长，当我们做决定的时候，我们先听取多个专家的意见，再做决定。
集成算法通常有两种方式，分别是投票选举（bagging）和再学习（boosting）。
投票选举的场景类似把专家召集到一个会议桌前，当做一个决定的时候，让 K 个专家（K 个模型）分别进行分类，然后选择出现次数最多的那个类作为最终的分类结果。
再学习相当于把 K 个专家（K 个分类器）进行加权融合，形成一个新的超级专家（强分类器），让这个超级专家做判断。
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier


# 加载数据
data = load_boston()
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)
# 使用 AdaBoost 回归模型
regressor = AdaBoostRegressor()
regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("房价预测结果 ", pred_y)
print("均方误差 = ", round(mse, 2))

# 使用决策树回归模型
dec_regressor = tree.DecisionTreeRegressor()
dec_regressor.fit(train_x, train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("决策树均方误差 = ", round(mse, 2))

# 使用 KNN 回归模型
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(train_x, train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN 均方误差 = ", round(mse, 2))

# AdaBoost 与决策树模型的比较
# 数据构造
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
# 从 12000 个数据中取前 2000 行作为测试集，其余作为训练集
train_x, train_y = X[2000:], y[2000:]
test_x, test_y = X[:2000], y[:2000]
# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(train_x, train_y)
dt_stump_err = 1.0 - dt_stump.score(test_x, test_y)
# 决策树分类器
dt = DecisionTreeClassifier()
dt.fit(train_x, train_y)
dt_err = 1.0 - dt.score(test_x, test_y)
# AdaBoost 分类器
# 设置 AdaBoost 迭代次数
n_estimators = 200
ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators)
ada.fit(train_x, train_y)
# 三个分类器的错误率可视化
fig = plt.figure()
# 设置 plt 正确显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
ax = fig.add_subplot(111)
ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-', label=u'决策树弱分类器 错误率')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label=u'决策树模型 错误率')
ada_err = np.zeros((n_estimators,))
# 遍历每次迭代的结果 i 为迭代次数, pred_y 为预测结果
for i, pred_y in enumerate(ada.staged_predict(test_x)):
    # 统计错误率
    ada_err[i] = zero_one_loss(pred_y, test_y)
# 绘制每次迭代的 AdaBoost 错误率
ax.plot(np.arange(n_estimators) + 1, ada_err, label='AdaBoost Test 错误率', color='orange')
ax.set_xlabel('迭代次数')
ax.set_ylabel('错误率')
leg = ax.legend(loc='upper right', fancybox=True)
plt.show()
