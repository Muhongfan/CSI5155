# ---- coding:UTF-8 ----
## read files
from os import path
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFromModel
import sklearn.model_selection as sk_model_selection

from sklearn.impute import SimpleImputer
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

## Read files to data frames
from sklearn.svm import LinearSVC
from sklearn.tree import tree


ams_df = pd.read_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/test_data.csv', index_col=0)


##### Modeling
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

ams_df['price'] = ams_df['price'].map(lambda x: np.log(x))

#y = ams_df.loc[:,'price']
#x = ams_df.loc.drop()

x = ams_df.drop('price', axis=1)
y = ams_df.price

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)



rf = RandomForestRegressor()

rf = RandomForestRegressor(n_estimators=500,
                               criterion='mse',
                               random_state=10,
                               n_jobs=-1,
                           max_depth=10)



rf.fit(X_train, y_train.astype('int'))
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
rmse_rf= (mean_squared_error(y_test,y_test_pred))**(1/2)

acc_tree = sk_model_selection.cross_val_score(rf, X_test, y_test, cv=10)

print(acc_tree.mean())

print('RMSE test: %.3f' % rmse_rf)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))
coefs_df = pd.DataFrame()
coefs_df['est_int'] = X_train.columns
coefs_df['coefs'] = rf.feature_importances_
print(coefs_df.sort_values('coefs', ascending=False).head(20))




# case1：学习曲线
# 构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
train_sizes, train_scores, test_scores = learning_curve(estimator=rf, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.01, 1.0])
plt.show()



'''


## after feature reduction
from sklearn.feature_selection import chi2
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter = 5000).fit(x, y.astype('int'))
model = SelectFromModel(lsvc, prefit=True)
X_new = pd.DataFrame(model.transform(x))
print(X_new.shape)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size = 0.2, random_state=1)

rf2 = RandomForestRegressor(n_estimators=500,
                               criterion='mse',
                               random_state=10,
                               n_jobs=-1)
rf2.fit(X_train_new, y_train_new)
y_train_pred2 = rf2.predict(X_train_new)
y_test_pred2 = rf2.predict(X_test_new)
rmse_rf2= (mean_squared_error(y_test_new,y_test_pred2))**(1/2)

acc_tree_2 = sk_model_selection.cross_val_score(rf2, X_test_new, y_test_new, cv=10)
print(acc_tree_2.mean())

print('RMSE test: %.3f' % rmse_rf2)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred2)))

coefs_df2 = pd.DataFrame()
coefs_df2['est_int'] = X_train_new.columns
coefs_df2['coefs'] = rf2.feature_importances_
print(coefs_df2.sort_values('coefs', ascending=False).head(20))

# case1：学习曲线
# 构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
train_sizes, train_scores, test_scores = learning_curve(estimator=rf2, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.01, 1.0])
plt.show()
'''
# DT regression
import pandas as pd
import numpy as np
#import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import sklearn.model_selection as sk_model_selection

regt = DecisionTreeRegressor()

#dot_data = tree.export_graphviz(regt, out_file=None)  # Export a decision tree in DOT format

#graph = graphviz.Source(dot_data)

#graph.render("tree")  # Save the source to file
regt.fit(X_train, y_train)
y_train_pred_regt = regt.predict(X_train)
y_test_pred_regt = regt.predict(X_test)
rmse_regt= (mean_squared_error(y_test,y_test_pred_regt))**(1/2)

acc_tree_regt = sk_model_selection.cross_val_score(regt, X_test, y_test, cv=10)

print(acc_tree_regt.mean())

print('RMSE test: %.3f' % rmse_regt)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred_regt)))
coefs_df_regt = pd.DataFrame()
coefs_df_regt['est_int'] = X_train.columns
coefs_df_regt['coefs'] = regt.feature_importances_
print(coefs_df_regt.sort_values('coefs', ascending=False).head(20))


# case1：学习曲线
# 构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
train_sizes, train_scores, test_scores = learning_curve(estimator=regt, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.01, 1.0])
plt.show()










