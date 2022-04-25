import seaborn as sns
import pandas as pd
import numpy as np

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
    X = X.drop(columns=['brand'])

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
metadata_df_clean = preprocess_data(metadata_df)

#Counter(" ".join(metadata_df_clean["description"]).split()).most_common(1000)

metadata_df_clean.to_csv('data/metadata_df_preprocessed'+category+'.csv',index=False)
df = pd.read_csv('data/metadata_df_preprocessed'+category+'.csv')
df = df.dropna(subset = ['description'])
df.to_csv('data/metadata_df_preprocessed_'+category+'.csv',index=False)