import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

## 包有更新，用这个
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

import matplotlib.pylab as plt


train = pd.read_csv('dataset/random_forest.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'

# print(train['Disbursed'].value_counts())


x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']
# print(X)

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))


# param_test1 = {'n_estimators':range(10,71,10)}
# ## 调参 第一次调参可以找到
# ## 弱训练器的最佳训练次数
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                                             min_samples_leaf=20,
#                                                             max_depth=8,
#                                                             max_features='sqrt' ,
#                                                            random_state=10),
#                        param_grid = param_test1,
#                         scoring='roc_auc',
#                         cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.grid_scores_, "\n",gsearch1.best_params_, "\n",gsearch1.best_score_,"\n")
#
# ## 调参
# param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
# gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
#                                                             min_samples_leaf=20,
#                                                            max_features='sqrt' ,
#                                                            oob_score=True,
#                                                            random_state=10),
#                         param_grid = param_test2,
#                         scoring='roc_auc',
#                         iid=False,
#                         cv=5)
# gsearch2.fit(X,y)
# print(gsearch2.grid_scores_, "\n", gsearch2.best_params_, "\n", gsearch2.best_score_ ,"\n")


print(rf0.predict([0,300000	,5	,20000,	1	,0	,37	,1,	1	,1	,1,	1	,0,	1	,1,	0	,1	,0	,0,	0	,0,	0	,0,	0,	0	,0,	0,	0,	0,	0	,0	,1	,0	,0	,0,	0,	0,	0	,0,	0	,0,	0,	0	,1,	1,	0	,1,	0	,0
]))