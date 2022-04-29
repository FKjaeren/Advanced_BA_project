import seaborn as sns
import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

def preprocess_data(metadata_df):
    
    X = metadata_df[['avg_rating','std_rating','num_ratings', 'category', 'also_buy', 'brand', 'rank','also_view', 'price','description']]

    # get number of also_buy
    X['also_buy'] = X['also_buy'].fillna('')
    X['also_buy'] = X['also_buy'].apply(get_number_also_buy)

    # get top 10 brands
    top = 10
    X['brand'] = X['brand'].str.replace('Unknown','')
    brands = X['brand'].value_counts().sort_values(ascending=False).index[0:(top)].to_list()
    X['top_brand'] = X['brand'].apply(lambda row: get_brand(row, brands))
    #X = X.drop(columns=['brand'])

    # sales rank information
    X['rank'] = X['rank'].apply(get_rank)
    X['rank'] = X['rank'].str.replace(',','')
    X['rank'] = X['rank'].str.extract('(\d+|$)')
    X['rank'] = pd.to_numeric(X['rank'], errors = 'coerce').fillna(0).apply(int)

    # Fill nan values for price data
    categories = []
    category_means = []
    categories = X.category.unique()
    for i in categories:
        temp = X[X['price'].isna()==False]
        mean_value = temp[temp['category']==i]['price'].mean()
        category_means.append(mean_value)

    dict = {'categories': categories,'category_means':category_means}

    category_stat_df = pd.DataFrame(dict)
    category_stat_df = category_stat_df.set_index('categories')

    X['price'] = X.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)

    # get dummies for: category, top_brands
    # get number of also_view
    X['also_view'] = X['also_view'].fillna('')
    X['also_view'] = X['also_view'].apply(get_number_also_buy)

    # drop nan's
    X = X.dropna(axis=0,subset=['avg_rating','num_ratings','category','description'])

    # get dummies for: category, top_brand
    #X = pd.get_dummies(X, columns=['category','top_brand'])

    #y = X['avg_rating']
    #X = X.drop(columns=['avg_rating'])
    return X

def preprocess_data(metadata_df):
    # split data, so we DON'T use test for preprocessing
    df = metadata_df[['avg_rating', 'std_rating','num_ratings', 'category', 'also_buy', 'brand', 'rank','also_view', 'price','description']]
    df = df.dropna(axis=0,subset=['avg_rating','num_ratings','category','description'])
    df_train, df_test = train_test_split(df, train_size=0.75)

    # get number of also_buy
    df_train['also_buy'] = df_train['also_buy'].fillna('').apply(get_number_also_buy)
    df_test['also_buy'] = df_test['also_buy'].fillna('').apply(get_number_also_buy)

    #top = 100
    #brands = df_train['brand'].value_counts().sort_values(ascending=False).index[0:(top)].to_list()
    #df_train['top_brand'] = df_train['brand'].apply(lambda row: get_brand(row, brands))
    #print("other brands",len(df_train[df_train['top_brand']=='Other']['top_brand']))
    #print("top brands",len(df_train[df_train['top_brand']==brands[0]]['top_brand']))
    #while (len(df_train[df_train['top_brand']=='Other']['top_brand']) >= df_train['top_brand'].value_counts()[1:].sum()):
    #    next_brand = df_train['brand'].value_counts().sort_values(ascending=False).index[top:top+10].to_list
    #    brands.append(next_brand)
    #    df_train['top_brand'] = df_train['brand'].apply(lambda row: get_brand(row, brands))
    #    top = top+11

    # sales rank information
    df_train['rank'] = df_train['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_train['rank'] = pd.to_numeric(df_train['rank'], errors = 'coerce').fillna(0).apply(int)
    df_test['rank'] = df_test['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_test['rank'] = pd.to_numeric(df_test['rank'], errors = 'coerce').fillna(0).apply(int)

    # Fill nan values for price data
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

    # get number of also_view
    df_train['also_view'] = df_train['also_view'].fillna('').apply(get_number_also_buy)
    df_test['also_view'] = df_test['also_view'].fillna('').apply(get_number_also_buy)

    # drop nan's
    # df_train = X.dropna(axis=0,subset=['avg_rating','num_ratings','category','description'])
    # df_test = X.dropna(axis=0,subset=['avg_rating','num_ratings','category','description'])

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

category = 'Candy & Chocolate'
metadata_df = pd.read_csv('data/df_'+category+'.csv')
# metadata_df_clean = preprocess_data(metadata_df)
df_train, df_test = preprocess_data(metadata_df)


#### Cluster brands:

brand_dummy_data = pd.get_dummies(df_train[['brand']], columns = ['brand'])


pca = PCA(n_components=100)
pca.fit(brand_dummy_data)
print(pca.explained_variance_ratio_)

pca = PCA(n_components=10)
pca_values = pca.fit_transform(brand_dummy_data)

dbscan_cluster = DBSCAN(eps=0.3, 
                        min_samples=1000)

# Fitting the clustering algorithm
dbscan_cluster.fit(brand_dummy_data)

# Adding the results to a new column in the dataframe
df_train['cluster'] = dbscan_cluster.labels_


#Counter(" ".join(metadata_df_clean["description"]).split()).most_common(1000)

#metadata_df_clean.to_csv('data/metadata_df_preprocessed'+category+'.csv',index=False)
df_train.to_csv('data/' + category + '/df_train.csv',index=False)
df_test.to_csv('data/' + category + '/df_test.csv',index=False)

# df = pd.read_csv('data/metadata_df_preprocessed'+category+'.csv')
df_train = pd.read_csv('data/' + category + '/df_train.csv')
df_test = pd.read_csv('data/' + category + '/df_test.csv')
df_train = df_train.dropna(subset = ['description'])
df_test = df_test.dropna(subset = ['description'])

df_train_subset = df_train[['avg_rating', 'num_ratings', 'category', 'also_buy', 'rank', 'also_view', 'price', 'description', 'brand']]


### Jeg kan ikke køre nedenstående
distance_matrix = gower.gower_matrix(df_train_subset)

# Configuring the parameters of the clustering algorithm
dbscan_cluster = DBSCAN(eps=0.3, 
                        min_samples=1000, 
                        metric="precomputed")

# Fitting the clustering algorithm
dbscan_cluster.fit(distance_matrix)

# Adding the results to a new column in the dataframe
df_train['cluster'] = dbscan_cluster.labels_

df_train.to_csv('data/metadata_df_preprocessed_'+category+'.csv',index=False)