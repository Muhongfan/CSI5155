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


ams_df= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_df_newone.csv'),index_col=0)
ott_df= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df.csv'),index_col=0)
ams_df_nostd= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_df_nostd.csv'),index_col=0)
ott_df_nostd= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_nostd.csv'),index_col=0)
ams_cat= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_cat.csv'),index_col=0)
ott_cat= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_cat.csv'),index_col=0)
ams_cat_ani = pd.read_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_cat_ani.csv',index_col=0)



##### Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#y = ams_df.loc[:,'price']
#x = ams_df.loc.drop()

x = ams_df.drop('price', axis=1)
y = ams_df.price
from sklearn.dummy.DummyRegressor import DummyRegressor
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=1)
rf = DummyRegressor()
rf.fit(X_train, y_train)
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




## after feature reduction
from sklearn.feature_selection import chi2
lsvc = LinearSVC(C=0.002, penalty="l1", dual=False, max_iter = 5000).fit(x, y)
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
print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))

coefs_df2 = pd.DataFrame()
coefs_df2['est_int'] = X_train_new.columns
coefs_df2['coefs'] = rf2.feature_importances_
print(coefs_df2.sort_values('coefs', ascending=False).head(20))