from CreateData import load_data
import seaborn as sns
import pandas as pd

def prepare_data(ratings_df, reviews_df, metadata_df):
    # create timestamps
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], origin = 'unix', unit = 's')
    reviews_df['timestamp'] = pd.to_datetime(reviews_df['unixReviewTime'], origin = 'unix', unit = 's')
    metadata_df['timestamp'] = pd.to_datetime(metadata_df['date'].apply(str), format = '%B %d, %Y', errors='coerce')

    # drop columns in reviews
    reviews_df = reviews_df.drop(columns=['unixReviewTime','reviewTime','reviewerName','vote','image','style','verified'])

    # drop columns in metadata
    metadata_df = metadata_df.drop(columns=['imageURL','imageURLHighRes'])
    
    # drop na's and duplicates
    reviews_df = reviews_df.dropna()
    reviews_df = reviews_df.drop_duplicates(keep='first')
    ratings_df = ratings_df.drop_duplicates(keep='first')

    # group ratings_df and merge with metadata
    grouped_ratings = ratings_df[['item','rating']].groupby(by='item').agg({'rating':'mean','item':'size'}).rename(columns={'rating':'avg_rating','item':'num_ratings'}).reset_index()
    metadata_df = grouped_ratings.merge(metadata_df, how='outer', left_on='item', right_on='asin')
    metadata_df['item'].fillna(metadata_df['asin'], inplace=True)
    metadata_df = metadata_df.drop(columns=['asin','date','tech1','tech2','fit'])

    # preprocess price
    metadata_df['price'] =  pd.to_numeric(metadata_df['price'].str.replace('$',''), errors='coerce')

    return reviews_df, metadata_df

def preprocess_data(metadata_df):
    
    X = metadata_df[['item', 'avg_rating', 'num_ratings', 'category', 'also_buy', 'brand', 'feature', 'rank',
       'also_view', 'main_cat', 'similar_item', 'price', 'details','timestamp']]

    # get category
    X['category'] = X['category'].fillna('')
    X['category'] = X['category'].apply(get_category)

    # get number of also_buy
    X['also_buy'] = X['also_buy'].fillna('')
    X['also_buy'] = X['also_buy'].apply(get_number_also_buy)

    # brand
    X['brand'] = X['brand'].str.replace('Unknown','')


    # get dummies for: category
    return

def get_category(row):
    if len(row) > 1:
        category = row[1]
    else:
        category = row
    return category

def get_number_also_buy(row):
    number = len(row)
    return number

def get_brand(row, **kwargs):
    if row in kwargs:
        return row
    else:
        return ''



rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json'
metadata_filepath = 'data/meta_Grocery_and_Gourmet_Food.json'

raw_ratings, raw_reviews, raw_metadata = load_data(rating_filepath=rating_filepath, review_filepath=review_filepath, metadata_filepath=metadata_filepath)

reviews_df, metadata_df = prepare_data(raw_ratings, raw_reviews, raw_metadata)


metadata_df['brand'] = metadata_df['brand'].str.replace('Unknown','')
brands = metadata_df['brand'].value_counts().sort_values(ascending=False).index[1:11].to_list()
metadata_df['new_brand'] = metadata_df['brand'].apply(get_brand, brands)

reviews_df.to_csv('data/reviews_df.csv',index=False)
metadata_df.to_csv('data/metadata_df.csv',index=False)







