import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# 生成2维正态分布，
# 生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
## 协方差必须是两个变量之间,两个变量都必须是一系列的点，有各自的均值
X1, y1 = make_gaussian_quantiles(
    cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
# 生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为1.5，
# 即X特征与Y特征的关系，不一定是正交
X2, y2 = make_gaussian_quantiles(
    mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)

## 关于协方差公式
## 一定注意，行列不能混淆
# X_test = np.array([[1,3,4,5],[2,6,2,2]])
## 具体公式怎么算的，就不推导了 E((X-U)(Y-λ))
##print(np.cov(X_test))
X = np.concatenate((X1, X2))
##　打乱顺序，将y的分类打乱
y = np.concatenate((y1, -y2 + 1 ))

#y = np.concatenate((y1, y2 ))
#print(X1.T)
#print(np.cov(X1.T))
#plt.plot(X1[:, 0], X1[:, 1])
#plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=y1)


#plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
#plt.show()

## adaboost分类器
bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         ## 200个弱分类器
                         n_estimators=200,
    ## 这个学习率是一个正则化的系数，为了防止过拟合
                        learning_rate=0.8)
## 拟合，用分类器给数据分类
## 这里的输出是bdt有了每个决策树的参数和描述
## 但是X和y是不变的
bdt.fit(X, y)


x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                     np.arange(x2_min, x2_max, 0.02))

y_direct = bdt.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_direct = y_direct.reshape(xx1.shape)

cs = plt.contourf(xx1, xx2, y_direct, cmap=plt.cm.Paired)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
##plt.show()



##查看一下拟合的分数
print("Score:", bdt.score(X,y))

