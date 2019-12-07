# ---- coding:UTF-8 ----
## read files
from os import path
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.impute import SimpleImputer
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

## Read files to data frames
from sklearn.svm import LinearSVC
from sklearn.tree import tree

ams_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/airbnb-amsterdam/listings_details.csv'), parse_dates=['last_scraped', 'first_review', 'last_review', 'host_since'])
ott_df = pd.read_csv(path.join('airbnb-ottawa/listings.csv'), parse_dates=['last_scraped', 'first_review', 'last_review', 'host_since'])
#print(sea_df.shape, bos_df.shape)

## hand-pick features of interest
cols = ['id', 'host_id', 'host_since', 'host_verifications', 'host_is_superhost', 'zipcode', 'bathrooms', 'bedrooms',
        'beds', 'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price', 'monthly_price',
        'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights','maximum_nights', 'calendar_updated',
        'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
        'number_of_reviews', 'first_review', 'last_review', 'review_scores_rating','review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture',
        'require_guest_phone_verification','calculated_host_listings_count', 'reviews_per_month']

ams_df = ams_df[cols]
ott_df = ott_df[cols]

#ams_df.set_index('id', inplace=True)
#print(ams_df.cleaning_fee.dtypes)
#print(ams_df.price.dtypes)
#print(ams_df.shape, ott_df.shape)

## check columns with missings in sea_df
temp1 = ams_df.isnull().sum() / ams_df.shape[0]
print(temp1[temp1 > 0])
## check columns with missings in bos_df
temp2 = ott_df.isnull().sum() / ott_df.shape[0]

## Let us drop features with more than 20% missings
ams_df = ams_df.drop(columns=temp1[temp1 > .2].index.values)
ott_df = ott_df.drop(columns=temp2[temp2 > .2].index.values)

print(temp1)

'''
## check the active listing
for col in ['first_review', 'last_review']:
        plt.figure(figsize=(8,4))
        sea_df[col].value_counts().plot(kind='bar')
        plt.title(col)
        plt.xticks(rotation=0)
        plt.show()
'''

print(ams_df.head(5))

# write the description of each features in txt
columns = ams_df.columns.values
des = open('/Users/momo/Documents/mhf/CSI5155/PRO/description.txt', 'a')
des.write(str(ams_df.describe().T))
des.close()


## Impute mode and mean to missings
imp_mode = SimpleImputer(strategy='most_frequent')
imp_median = SimpleImputer(strategy='median')
imp_mean = SimpleImputer(strategy='mean')

ams_df[['host_is_superhost', 'zipcode','cleaning_fee']] = imp_mode.fit_transform(ams_df[['host_is_superhost', 'zipcode', 'cleaning_fee']])
ott_df[['host_is_superhost', 'zipcode','cleaning_fee']] = imp_mode.fit_transform(ott_df[['host_is_superhost', 'zipcode', 'cleaning_fee']])

ams_df[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']] = imp_median.fit_transform(ams_df[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']])
ott_df[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']] = imp_median.fit_transform(ott_df[['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']])

ams_df[['bathrooms', 'bedrooms', 'beds', 'reviews_per_month','host_days','first_review_days','last_review_days']] = imp_mean.fit_transform(ams_df[['bathrooms', 'bedrooms', 'beds', 'reviews_per_month','host_days','first_review_days','last_review_days']])
ott_df[['bathrooms', 'bedrooms', 'beds', 'reviews_per_month','host_days','first_review_days','last_review_days']] = imp_mean.fit_transform(ott_df[['bathrooms', 'bedrooms', 'beds', 'reviews_per_month','host_days','first_review_days','last_review_days']])

# Transforming binary features
binary = {'f' : 0,'t' : 1}
ams_df['require_guest_profile_picture'] = ams_df['require_guest_profile_picture'].map(binary)
ams_df['require_guest_phone_verification'] = ams_df['require_guest_phone_verification'].map(binary)
ams_df['instant_bookable'] = ams_df['instant_bookable'].map(binary)
ams_df['host_is_superhost'] = ams_df['host_is_superhost'].map(binary)
ams_df['has_availability'] = ams_df['has_availability'].map(binary)

ott_df['require_guest_profile_picture'] = ott_df['require_guest_profile_picture'].map(binary)
ott_df['require_guest_phone_verification'] = ott_df['require_guest_phone_verification'].map(binary)
ott_df['instant_bookable'] = ott_df['instant_bookable'].map(binary)
ott_df['host_is_superhost'] = ott_df['host_is_superhost'].map(binary)
ott_df['has_availability'] = ott_df['has_availability'].map(binary)

#print(ams_df.info())

# Transforming certain features¶
## 1) Zipcode: remove unnecessary details or typos
## extract numbers

ams_df.zipcode[ams_df.zipcode.astype(str).str.contains('[^0-9]', regex=True)] = ams_df.zipcode.astype(str).str.extract(r'(?:[^0-9])(\d+)')[0].value_counts().index[0]
#ott_df.zipcode = ott_df.zipcode.astype(str)[0:3]
#print('zip code:',ams_df.zipcode.value_counts(normalize=True))

## 2) host_since\first reviews\ last_reviews: change to days (difference of last day of dataframe and host_since)
## Get last day as baseline point
sea_now = ams_df.host_since.max()
bos_now = ott_df.host_since.max()
## retrieve days of difference from basline date
ams_df['host_days'] = ams_df.host_since.map(lambda x: (sea_now - x).days)
ott_df['host_days'] = ott_df.host_since.map(lambda x: (bos_now - x).days)
## drop host_since column
ams_df = ams_df.drop(columns=['host_since'])
ott_df = ott_df.drop(columns=['host_since'])

#first review
ams_df['first_review_days'] = ams_df.first_review.map(lambda x: (sea_now - x).days)
ott_df['first_review_days'] = ott_df.first_review.map(lambda x: (bos_now - x).days)
ams_df = ams_df.drop(columns=['first_review'])
ott_df = ott_df.drop(columns=['first_review'])
#last reveiew
ams_df['last_review_days'] = ams_df.last_review.map(lambda x: (sea_now - x).days)
ott_df['last_review_days'] = ott_df.last_review.map(lambda x: (bos_now - x).days)
ams_df = ams_df.drop(columns=['last_review'])
ott_df = ott_df.drop(columns=['last_review'])


## 3) Price & extra_people: str -> float
## drop '$' and ','
ams_df.price = ams_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
ott_df.price = ott_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)


ams_df.cleaning_fee = ams_df.cleaning_fee.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
#ott_df.cleaning_fee = ott_df.cleaning_fee.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)

ams_df.extra_people = ams_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
ott_df.extra_people = ott_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)

## 4) host_verifications
### seattle
ams_df.host_verifications = ams_df.host_verifications.replace(['[]', 'None'], "['none']")
### define seattle categorical dummy dataframe
ams_cat_ver = pd.get_dummies(ams_df.host_verifications.map(literal_eval).apply(pd.Series).stack(), prefix='host_ver').sum(level=0)
#print(ams_cat.columns)
### boston
ott_df.host_verifications = ott_df.host_verifications.replace(['[]', 'None'], "['none']")
### define boston categorical dummy dataframe
ott_cat_ver = pd.get_dummies(ott_df.host_verifications.map(literal_eval).apply(pd.Series).stack(), prefix='host_ver').sum(level=0)

#print('original df shape', ams_df.shape, ott_df.shape)
#print('cat shape', ams_cat.shape, ott_cat.shape)

### unify column width and order
#bos_cat['host_ver_photographer'] = 0
#ams_cat_ver = ams_cat_ver[ott_cat_ver.columns]
## drop host_verifications from sea_df, bos_df
ams_df = ams_df.drop(columns=['host_verifications'])
ott_df = ott_df.drop(columns=['host_verifications'])

#print('after drop host_vertifications', ams_df.shape, ott_df.shape)
#print('after drop host_vertifications', ams_cat.shape, ott_cat.shape)

## 5) amenities
### change format to string list same as host_verifications
ams_df['amenities'] = ams_df['amenities'].map(lambda d: [amenity.replace('"', "").replace("{", "").replace("}", "") for amenity in d.split(",")]).astype(str)
ott_df['amenities'] = ott_df['amenities'].map(lambda d: [amenity.replace('"', "").replace("{", "").replace("}", "") for amenity in d.split(",")]).astype(str)

### seattle
ams_df.amenities = ams_df.amenities.replace("['']", "['none']")
### boston
ott_df.amenities = ott_df.amenities.replace("['']", "['none']")
## temporary dataframe before adding to sea_cat
ams_cat_ame = pd.get_dummies(ams_df.amenities.map(literal_eval).apply(pd.Series).stack(), prefix='amenities').sum(level=0)
ams_cat_ani = temp1
## temporary dataframe before adding to bos_cat
#print('temp1:',temp1)
ott_cat_ame = pd.get_dummies(ott_df.amenities.map(literal_eval).apply(pd.Series).stack(), prefix='amenities').sum(level=0)
#print('temp2.shape',temp2.shape)
### unify column width and order
colus = ['amenities_Free Parking on Street',
        'amenities_Paid Parking Off Premises',
        'amenities_translation missing: en.hosting_amenity_49',
        'amenities_translation missing: en.hosting_amenity_50']
for col in colus:
    temp1[col] = 0
# drop the features that are not in both dataset
con_item = []
con_item2 =[]
for item in ott_cat_ame.columns.values:
    if item not in ams_cat_ame.columns.values:
        con_item.append(item)
#print(len(con_item))

for item2 in ams_cat_ame.columns.values:
    if item2 not in ott_cat_ame.columns.values:
        con_item2.append(item2)
#print(con_item2)
ams_cat_ame = ams_cat_ame.drop(columns = con_item2)
ott_cat_ame = ott_cat_ame.drop(columns = con_item)
#print(temp1.shape)
#print(temp2.shape)
#ott_cat_ame.columns = ams_cat_ame.columns

## concatenate dummy variable dataframes
ams_cat = pd.concat([ams_cat_ver, ams_cat_ame], axis=1)
ott_cat = pd.concat([ott_cat_ver, ott_cat_ame], axis=1)
print('ams_cat shape', ams_cat.shape)
## drop amenities from sea_df, bos_df
ams_df = ams_df.drop(columns=['amenities'])
ott_df = ott_df.drop(columns=['amenities'])


## 6) bed type
print(ams_df.bed_type.value_counts())
ams_df = ams_df.drop(columns=['bed_type'])

## 7）calendar updated
print(ams_df.calendar_updated.value_counts())
ams_df['calendar_updated'] = ams_df['calendar_updated'].map(lambda d: [calendar_updated.replace('"', "").replace("{", "").replace("}", "") for calendar_updated in d.split(",")]).astype(str)
ott_df['calendar_updated'] = ott_df['calendar_updated'].map(lambda d: [calendar_updated.replace('"', "").replace("{", "").replace("}", "") for calendar_updated in d.split(",")]).astype(str)

### seattle
ams_df.calendar_updated = ams_df.calendar_updated.replace("['']", "['none']")
### boston
ott_df.calendar_updated = ott_df.calendar_updated.replace("['']", "['none']")
## temporary dataframe before adding to sea_cat
temp_cal = pd.get_dummies(ams_df.calendar_updated.map(literal_eval).apply(pd.Series).stack(), prefix='calendar_updated').sum(level=0)
ams_df = ams_df.drop(columns=['calendar_updated'])
ams_df = pd.concat([ams_df, temp_cal], axis=1)

## 8）cancellation_policy
print(ams_df.cancellation_policy.value_counts())
ams_df['cancellation_policy'] = ams_df['cancellation_policy'].map(lambda d: [cancellation_policy.replace('"', "").replace("{", "").replace("}", "") for cancellation_policy in d.split(",")]).astype(str)
ott_df['cancellation_policy'] = ott_df['cancellation_policy'].map(lambda d: [cancellation_policy.replace('"', "").replace("{", "").replace("}", "") for cancellation_policy in d.split(",")]).astype(str)

### seattle
ams_df.cancellation_policy = ams_df.cancellation_policy.replace("['']", "['none']")
### boston
ott_df.cancellation_policy = ott_df.cancellation_policy.replace("['']", "['none']")
## temporary dataframe before adding to sea_cat
temp_can = pd.get_dummies(ams_df.cancellation_policy.map(literal_eval).apply(pd.Series).stack(), prefix='cancellation_policy').sum(level=0)
ams_df = ams_df.drop(columns=['cancellation_policy'])
ams_df = pd.concat([ams_df, temp_can], axis=1)

## 9) drop column that contains one value
print(ams_df.has_availability.value_counts())
'''
name=[]
for n in ams_df.columns.values:
    name = ams_df.select_dtypes(include=[int, float]).columns.values
for i in range(len(name)):
    ams_df.hist(name[i])
    plt.show()
#for i in range(len(name)):
#    ams_df.hist(name[i])
'''
ams_df = ams_df.drop(columns=['has_availability'])
ott_df = ott_df.drop(columns=['has_availability'])

# sum of missing values after dropping

ams_sum_null = ams_df.isnull().sum()

print('After imputed',ams_sum_null)




'''

# Calibration for review scores
def bin_column(col, bins, labels, na_label='unknown'):

    ams_df[col] = pd.cut(ams_df[col], bins=bins, labels=labels, include_lowest=True)
    ams_df[col] = ams_df[col].astype('str')
    ams_df[col].fillna(na_label, inplace=True)


variables_to_plot = list(ams_df.columns[ams_df.columns.str.startswith("review_scores") == True])
# Binning for all columns scored out of 10
variables_to_plot.pop(0)

for col in variables_to_plot:
    bin_column(col,
               bins=[0, 8, 9, 10],
               labels=['0-8/10', '9/10', '10/10'],
               na_label='no reviews')

# Binning review_scores_rating since it scored out of 100
bin_column('review_scores_rating',
           bins=[0, 80, 95, 100],
           labels=['0-79/100', '80-94/100', '95-100/100'],
           na_label='no reviews')


'''
# Processing numerical variables - Standardization
#print(ams_df.columns)
## Retrieve numerical features
ams_num_no_std = ams_df.select_dtypes(include=[int, float]).drop(columns=['id', 'host_id', 'price'])
ott_num_no_std = ott_df.select_dtypes(include=[int, float]).drop(columns=['id', 'host_id', 'price'])
#print('std:', ams_num_no_std.columns)
## make copy
sea_num = ams_num_no_std.copy()
bos_num = ott_num_no_std.copy()

## standardizing
scaler = StandardScaler()
sea_num[sea_num.columns] = scaler.fit_transform(ams_num_no_std.astype(np.float))
bos_num[bos_num.columns] = scaler.fit_transform(ott_num_no_std.astype(np.float))

## putting it all together (df, num, cat)
ams_df = pd.concat([ams_df[['id', 'host_id', 'zipcode', 'price']], sea_num], axis=1)
ott_df = pd.concat([ott_df[['id', 'host_id', 'zipcode', 'price']], bos_num], axis=1)

## for EDA, getting not standardized numericals
ams_df_nostd = pd.concat([ams_df[['id', 'host_id', 'zipcode', 'price']], ams_num_no_std], axis=1)
ott_df_nostd = pd.concat([ott_df[['id', 'host_id', 'zipcode', 'price']], ott_num_no_std], axis=1)
#print(ott_df_nostd.shape)
#print(ott_df.shape)
#ams_df = pd.concat([ams_df, ams_cat], axis=1)

#print(ams_df.head(1))
#print(ams_df.info())
# Processing categorical variables

ams_df.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df_new.csv')
ott_df.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_e.csv')
ams_cat_ani.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ams_cat_ani.csv')
ams_df_nostd.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df_nostd_new.csv')
ott_df_nostd.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_df_nostd.csv')
ams_cat.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_cat_new.csv')
ott_cat.to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/dataset/ott_cat_e.csv')


