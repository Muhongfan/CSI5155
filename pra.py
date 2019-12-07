# ---- coding:UTF-8 ----
## read files
from os import path
import pandas as pd
from sklearn.impute import SimpleImputer
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

## Read files to data frames
sea_df = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df_new.csv'))
bos_df = pd.read_csv(path.join('airbnb-ottawa/listings.csv'), parse_dates=['last_scraped', 'first_review', 'last_review', 'host_since'])
nostd = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df_nostd_new.csv'))
nostd.head(5000).to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/test_nostd_new.csv')
sea_df.head(5000).to_csv('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/test_data.csv')
ams = pd.read_csv(path.join('/Users/momo/Documents/mhf/CSI5155/PRO/New_dataset/ams_df.csv'))
print('w')