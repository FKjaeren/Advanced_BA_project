import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split


def preprocess_data(df_train, df_test):
    # split data, so we DON'T use test for preprocessing
    # get number of also_buy
    df_train['also_buy'] = df_train['also_buy'].fillna('').apply(get_number_also_buy)
    df_test['also_buy'] = df_test['also_buy'].fillna('').apply(get_number_also_buy)

    # sales rank information
    df_train['rank'] = df_train['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_train['rank'] = pd.to_numeric(df_train['rank'], errors = 'coerce').fillna(0).apply(int)
    df_test['rank'] = df_test['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_test['rank'] = pd.to_numeric(df_test['rank'], errors = 'coerce').fillna(0).apply(int)

    # Fill nan values for price data
    # get number of also_view
    df_train['also_view'] = df_train['also_view'].fillna('').apply(get_number_also_buy)
    df_test['also_view'] = df_test['also_view'].fillna('').apply(get_number_also_buy)

    return df_train, df_test

def get_number_also_buy(row):
    number = len(row)
    return number

def get_brand(row, brands):
    if row in brands:
        return row
    else:
        return 'Other'

def get_rank(row):
    if isinstance(row, list):
        if len(row) > 0:
            return row[0]
        else:
            return ''
    else:
        return row

def preprocess_price(metadata_df):
    df = metadata_df.drop(columns = ['item','title','feature','main_cat','similar_item','details','timestamp'])
    
    df = df.dropna(axis=0,subset=['avg_rating','num_ratings','description'])
    df_train, df_test = train_test_split(df, train_size=0.75)
    categories = []
    category_means = []
    categories = df_train.category.unique()
    for i in categories:
        temp = df_train[df_train['price'].isna() == False]
        mean_value = temp[temp['category'] == i]['price'].mean()
        category_means.append(mean_value)
    dict = {'categories': categories,'category_means': category_means}
    category_stat_df = pd.DataFrame(dict)
    category_stat_df = category_stat_df.set_index('categories')

    df_train['price'] = df_train.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)
    df_test['price'] = df_test.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)
    columns = df_train.columns
    if 'category' in columns:
        df_train = df_train.drop(columns = ['category'])
        df_test = df_test.drop(columns = ['category'])
    if 'orig category' in columns:
        df_train = df_train.drop(columns = ['orig category'])
        df_test = df_test.drop(columns = ['orig category'])
    return df_train, df_test

category = 'Candy & Chocolate'
metadata_df = pd.read_csv('data/'+category+'/df_'+category+'.csv')
metadata_df['orig category'] = metadata_df['category']
dummy_df = pd.get_dummies(metadata_df, columns=['brand','orig category'])
metadata_df = metadata_df.drop(columns=['brand'])

df_train, df_test = preprocess_price(metadata_df)
df_train_dummy, df_test_dummy = preprocess_price(dummy_df)

df_train, df_test = preprocess_data(df_train, df_test)
df_train_dummy, df_test_dummy = preprocess_data(df_train_dummy, df_test_dummy)


df_train_dummy = df_train_dummy.drop(columns = ['description','std_rating'])
df_test_dummy = df_test_dummy.drop(columns = ['description','std_rating'])

pca = PCA(n_components=100)
pca.fit(df_train_dummy)
print(pca.explained_variance_ratio_)
pca = PCA(n_components=2).fit(df_train_dummy)
pca_values = pca.fit_transform(df_train_dummy)
kmeans = KMeans().fit(pca_values)
df_train['cluster'] = kmeans.labels_
pca_values_test = pca.transform(df_test_dummy)
df_test['cluster'] = kmeans.predict(pca_values_test)

df_train = pd.get_dummies(df_train, columns = ['cluster'])
df_test = pd.get_dummies(df_test, columns = ['cluster'])

#sns.scatterplot('price','num_ratings', hue = 'cluster', data = df_train)
#plt.show()

df_train.to_csv('data/' + category + '/df_train_preprocessed.csv',index=False)
df_test.to_csv('data/' + category + '/df_test_preprocessed.csv',index=False)
