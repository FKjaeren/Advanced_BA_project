from CreateData import load_data
import seaborn as sns
import pandas as pd

def preprocess_data(ratings_df, reviews_df, metadata_df):
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
    metadata_df = metadata_df.drop(columns=['asin','date'])

    # preprocess columns
    metadata_df['price'] =  pd.to_numeric(metadata_df['price'].str.replace('$',''), errors='coerce')
   

    # merge dfs
    # df = ratings_df.merge(reviews_df, how='outer', left_on='item', right_on='asin')
    # df = df.merge(metadata_df, how='outer', left_on='item', right_on='asin')

    return reviews_df, metadata_df

rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json'
metadata_filepath = 'data/meta_Grocery_and_Gourmet_Food.json'

raw_ratings, raw_reviews, raw_metadata = load_data(rating_filepath=rating_filepath, review_filepath=review_filepath, metadata_filepath=metadata_filepath)

reviews_df, metadata_df = preprocess_data(raw_ratings, raw_reviews, raw_metadata)


reviews_df.to_csv('data/reviews_df.csv')
metadata_df.to_csv('data/metadata_df.csv')







