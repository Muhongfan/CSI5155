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
from sklearn.dummy import DummyRegressor

x = ams_df.drop('price', axis=1)
y = ams_df.price
from sklearn.dummy import DummyRegressor
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
rf = DummyRegressor(strategy = 'mean')
rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
rmse_rf= (mean_squared_error(y_test,y_test_pred))**(1/2)

acc_tree = sk_model_selection.cross_val_score(rf, X_test, y_test, cv=10)

print(acc_tree.mean())

print('RMSE test: %.3f' % rmse_rf)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))



## after feature reduction
from sklearn.feature_selection import chi2
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter = 5000).fit(x, y.astype('int'))
model = SelectFromModel(lsvc, prefit=True)
X_new = pd.DataFrame(model.transform(x))
print(X_new.shape)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size = 0.2, random_state=1)

rf2 = DummyRegressor(strategy = 'mean')
rf2.fit(X_train_new, y_train_new)
y_train_pred2 = rf2.predict(X_train_new)
y_test_pred2 = rf2.predict(X_test_new)
rmse_rf2= (mean_squared_error(y_test_new,y_test_pred2))**(1/2)

acc_tree_2 = sk_model_selection.cross_val_score(rf2, X_test_new, y_test_new, cv=10)
print(acc_tree_2.mean())

print('RMSE test: %.3f' % rmse_rf2)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))

coefs_df2 = pd.DataFrame()
coefs_df2['est_int'] = X_train_new.columns


train_sizes, train_scores, test_scores = learning_curve(estimator=rf, X=X_train, y=y_train,
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
