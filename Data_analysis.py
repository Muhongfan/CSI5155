# ---- coding:UTF-8 ----

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
## read files
from os import path
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.impute import SimpleImputer
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

## Read files to data frames
from sklearn.svm import LinearSVC
from sklearn.tree import tree

#ams_df= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_df_5000.csv'),index_col=0)
ams_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/test_data.csv'), index_col=0)

ott_df= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df.csv'),index_col=0)
ams_df_nostd= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df_nostd_new.csv'),index_col=0)
ott_df_nostd= pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_nostd.csv'),index_col=0)




ams_price = ams_df_nostd.price.copy()
ott_price = ott_df_nostd.price.copy()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ams_price.hist(bins=30, color='blue', alpha=0.4, ax=ax1)
ott_price.hist(bins=30, color='red', alpha=0.4, ax=ax2)
sns.despine(top=True, right=True, left=True)

# plot price
ax1.set_title('Amsterdam Price')
ax2.set_title('Ottawa Price')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Price')
ax2.set_xlabel('Price')
plt.show()
plt.clf()


ams_price_min = []
for item in ams_price:
    if item != 0.0:
     ams_price_min.append(item)
print('length of sea_price_min',len(ams_price_min))
ams_price_min = pd.Series(ams_price_min)
print('min sea_price', ams_price_min.min())
print('max sea_price', ams_price_min.max())

ott_price_min = []
for item in ott_price:
    if item != 0.0:
     ott_price_min.append(item)
print('length of bos_price_min',len(ott_price_min))
ott_price_min = pd.Series(ott_price_min)

print('num of bos_price=:',(ott_price == 0.0).sum())
## Transform price to log-price
ams_price_log = ams_price_min.map(lambda x: np.log(x))
ott_price_log = ott_price_min.map(lambda x: np.log(x))

## Visualize again
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ams_price_log.hist(bins=50, color='blue', alpha=0.4, ax=ax3)
ott_price_log.hist(bins=40, color='red', alpha=0.4, ax=ax4)
sns.despine(top=True, right=True, left=True)

ax3.set_title('Amsterdam Log-price')
ax4.set_title('Ottawa Log-price')
ax3.set_ylabel('Frequency')
ax3.set_xlabel('log( Price )')
ax4.set_xlabel('log( Price )')
plt.show()
plt.clf()


## display statistics
_ = pd.concat([ams_price.describe(), ott_price.describe()], axis=1)
_.columns = ['Amsterdam log-price', 'Ottawa log-price']
print(_)

#print(sea_df.shape, bos_df.shape)
colus =  [i for i in ams_df_nostd.columns.values]
print(colus)


## cor relation
#col = ['calculated_host_listings_count', 'minimum_nights', 'bathrooms', 'bedrooms', 'beds', 'price', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month']
sns.set(style="ticks", color_codes=True)
sns.pairplot(ams_df_nostd.loc[(ams_df_nostd.price <= 600) & (ams_df_nostd.price > 0)][colus].dropna())
plt.show()
plt.clf()

corr = ams_df_nostd.loc[(ams_df_nostd.price <= 600) & (ams_df_nostd.price > 0)][colus].dropna().corr()
plt.figure(figsize = (150,150))
sns.set(font_scale=1)
sns.heatmap(corr, cbar = True, annot=True, square = True, fmt = '.2f', xticklabels=colus, yticklabels=colus)
plt.show()
plt.clf()


'''

def binary_count_and_price_plot(col, figsize=(6,6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(col, fontsize=16, y=1)
    plt.subplots_adjust(top=0.80)  # So that the suptitle does not overlap with the ax plot titles

    ams_cat.groupby(col).size().plot(kind='bar', ax=ax1, color=['pink', 'blue'])
    ax1.set_xticklabels(labels=['false', 'true'], rotation=0)
    ax1.set_title('Category count')
    ax1.set_xlabel('')

    ams_cat.groupby(col).price.median().plot(kind='bar', ax=ax2, color=['pink', 'blue'])
    ax2.set_xticklabels(labels=['false', 'true'], rotation=0)
    ax2.set_title('Median price ($)')
    ax2.set_xlabel('')
    #plt.savefig('/Users/momo/Documents/mhf/CSI5155/PRO/picture/%s.jpg' % (col))
    plt.show()


# EDA of catigrical features
ams_cat = pd.concat([ams_df.id,ams_df.price, ams_cat], axis=1)
ams_cat_columns = ams_cat.iloc[:,:].columns
for col in ams_cat_columns:
    binary_count_and_price_plot(col)
    plt.savefig('/Users/momo/Documents/mhf/CSI5155/PRO/picture/%s.jpg' % (col))
    

colus =  [i for i in ams_cat_ani.columns.values]
ams_cat_ani = pd.concat([ams_df.id,ams_df.price, ams_cat_ani], axis=1)

corr2 = ams_cat_ani.loc[(ams_cat_ani.price <= 600) & (ams_cat_ani.price > 0)][colus].dropna().corr()
plt.figure(figsize = (60,60))
sns.set(font_scale=1)
sns.heatmap(corr2, cbar = True, annot=True, square = True, fmt = '.2f', xticklabels=colus, yticklabels=colus)
plt.show()
plt.clf()


# Replacing columns with f/t with 0/1
#ams_df.replace({'f': 0, 't': 1}, inplace=True)



'''
