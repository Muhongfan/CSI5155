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


ams_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/test_data.csv'), index_col=0)
ams_df['price'] = ams_df['price'].map(lambda x: np.log(x))



from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

x = ams_df.drop('price', axis=1)
y = ams_df.price

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#使用训练数据进行参数估计
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
rmse_lr= (mean_squared_error(y_test,y_test_pred))**(1/2)
acc_lr = sk_model_selection.cross_val_score(lr, X_test, y_test, cv=10)

print(acc_lr.mean())
print('RMSE test: %.3f' % rmse_lr)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))

coefs_df = pd.DataFrame()
coefs_df['est_int'] = X_train.columns
coefs_df['coefs'] = lr.coef_
print(coefs_df.sort_values('coefs', ascending=False).head(20))


'''
# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) #设置最多添加几次幂的特征项
poly.fit(x)
x2 = poly.transform(x)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(x2, y, test_size = 0.2, random_state=1)


lr_poly=PolynomialFeatures()
#使用训练数据进行参数估计
lr_poly.fit(X_train_poly,y_train_poly)
y_train_pred_poly = lr_poly.predict(X_train_poly)
y_test_pred_poly = lr_poly.predict(X_test_poly)
rmse_lr_poly= (mean_squared_error(y_test_poly,y_test_pred_poly))**(1/2)
acc_lr_poly = sk_model_selection.cross_val_score(lr_poly, X_test_poly, y_test_poly, cv=10)

print(acc_lr_poly.mean())
print('RMSE test: %.3f' % rmse_lr_poly)
print('R^2 test: %.3f' % (r2_score(y_test_poly, y_test_pred_poly)))



coefs_df_poly = pd.DataFrame()
coefs_df_poly['est_int'] = x2.columns
coefs_df_poly['coefs'] = lr_poly.coef_
print(coefs_df_poly.sort_values('coefs', ascending=False).head(20))

# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lr_poly, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

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


# Lasso Regression
from sklearn.linear_model import ElasticNet
lg = ElasticNet()
lg.fit(X_train,y_train)
y_train_pred_lg = lg.predict(X_train)
y_test_pred_lg = lg.predict(X_test)
rmse_lg= (mean_squared_error(y_test,y_test_pred_lg))**(1/2)
acc_lg = sk_model_selection.cross_val_score(lg, X_test, y_test, cv=10)

print(acc_lg.mean())
print('RMSE test: %.3f' % rmse_lg)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred_lg)))

coefs_df_lg= pd.DataFrame()
coefs_df_lg['est_int'] = X_train.columns
coefs_df_lg['coefs'] = lg.coef_
print(coefs_df_lg.sort_values('coefs', ascending=False).head(20))

# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=lg, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

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

