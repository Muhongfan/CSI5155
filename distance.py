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

##### Modeling
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#y = ams_df.loc[:,'price']
#x = ams_df.loc.drop()

x = ams_df.drop('price', axis=1)
y = ams_df.price

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)


# KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]

    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


knn = KNeighborsRegressor()
knn.fit(X_train, y_train.ravel())
y_train_pred_knn = knn.predict(X_train)
y_test_pred_knn = knn.predict(X_test)



rmse_knn = (mean_squared_error(y_test,y_test_pred_knn))**(1/2)
acc_knn = sk_model_selection.cross_val_score(knn, X_test, y_test, cv=10)

print(acc_knn.mean())
print('RMSE test: %.3f' % rmse_knn)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred_knn)))

'''
coefs_df = pd.DataFrame()
coefs_df['est_int'] = X_train.columns

'''



# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=knn, X=X_train, y=y_train,
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

# SVR
from sklearn.svm import SVR
linear_svr = SVR()
linear_svr.fit(X_train, y_train.ravel())
y_train_pred = linear_svr.predict(X_train)
y_test_pred = linear_svr.predict(X_test)

rmse = (mean_squared_error(y_test,y_test_pred))**(1/2)

acc = sk_model_selection.cross_val_score(linear_svr, X_test, y_test, cv=10)

print(acc.mean())

print('RMSE test: %.3f' % rmse)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))


# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=linear_svr, X=X_train, y=y_train,
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


plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
#预测值与偏差的关系
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
#plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()
# plt.savefig('./figures/slr_residuals.png', dpi=300)
plt.show()