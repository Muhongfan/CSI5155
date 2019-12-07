# ---- coding:UTF-8 ----
## read files
from os import path
import pandas as pd
from imblearn.pipeline import Pipeline
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

ams_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_df_e.csv'), index_col=0)
ott_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_e.csv'), index_col=0)
ams_df_nostd = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_df_nostd.csv'), index_col=0)
ott_df_nostd = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_nostd.csv'), index_col=0)
ams_cat = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_cat_e.csv'), index_col=0)
ott_cat = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_cat_e.csv'), index_col=0)
ams_cat_ani = pd.read_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_cat_ani.csv', index_col=0)

#print(ams_df.columns)
##### Modeling
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# y = ams_df.loc[:,'price']
# x = ams_df.loc.drop()

x = ams_df.drop('price', axis=1)
y = ams_df.price


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter = 5000).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = pd.DataFrame(model.transform(x))
print(X_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.1, random_state=1)

from sklearn.ensemble import GradientBoostingRegressor
#vr = GradientBoostingRegressor(n_estimators =40, min_samples_leaf =160,max_features=36,max_depth= 192)
gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)
rmse_vr = (mean_squared_error(y_test, y_test_pred)) ** (1 / 2)
acc_vr = sk_model_selection.cross_val_score(gbr, X_test, y_test, cv=10)
print(acc_vr.mean())
print('RMSE test: %.3f' % rmse_vr)
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))

coefs_df = pd.DataFrame()
coefs_df['est_int'] = X_train.columns
coefs_df['coefs'] = gbr.feature_importances_
print(coefs_df.sort_values('coefs', ascending=False).head(20))

'''
## Adjust
parameters = {#'n_estimators': [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]}
              #'min_samples_leaf': [ 3, 10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,300,400,500]}
              #'alpha': [0.1, 0.3, 0.6, 0.9]
               #'max_features':[30,32,34,36,38,40,42,44,46,48,50]
                'max_depth':[i for i in range(10,200)]  }  # 定义要优化的参数信息
model_gs = GridSearchCV(estimator=vr, param_grid=parameters, cv=10)
model_gs.fit(X_train,y_train)
print(model_gs.best_params_, model_gs.best_score_)
'''


# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=gbr, X=X_train, y=y_train,
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