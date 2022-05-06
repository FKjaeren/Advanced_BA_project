# Import packages
import seaborn as sns
import pandas as pd
import numpy as np


### Import raw data
def load_data(rating_filepath, review_filepath, metadata_filepath):
    ratings_df = pd.read_csv(rating_filepath, names = ['item','user','rating','timestamp'])
    reviews_df = pd.read_json(review_filepath, lines=True)
    metadata_df = pd.read_json(metadata_filepath, lines=True)
    return ratings_df, reviews_df, metadata_df

### Function for 
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

    # group ratings_df and merge with metadata, so there is one dataframe with both ratings and information of products
    grouped_ratings = ratings_df[['item','rating']].groupby(by='item').agg({'rating':['mean','std'],'item':'size'}).rename(columns={'statistics':'avg_rating','item':'num_ratings'}).reset_index()
    grouped_ratings.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped_ratings.columns.values]
    grouped_ratings = grouped_ratings.rename(columns = {'rating_mean':'avg_rating','rating_std':'std_rating','num_ratings_size':'num_ratings'})
    metadata_df = grouped_ratings.merge(metadata_df, how='outer', left_on='item', right_on='asin')
    metadata_df['item'].fillna(metadata_df['asin'], inplace=True)
    metadata_df = metadata_df.drop(columns=['asin','date','tech1','tech2','fit'])

    # preprocess price
    metadata_df['price'] =  pd.to_numeric(metadata_df['price'].str.replace('$',''), errors='coerce')

    # Fill nan with empty space and use the get_category function
    metadata_df['category'] = metadata_df['category'].fillna('')
    metadata_df['category'] = metadata_df['category'].apply(get_category)
    

    return reviews_df, metadata_df

# Function to return only the first name in each category variable.
def get_category(row):
    if len(row) > 1:
        category = row[1]
    else:
        category = row
    return category


rating_filepath = 'raw_data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'raw_data/Grocery_and_Gourmet_Food_5.json' 
metadata_filepath = 'raw_data/meta_Grocery_and_Gourmet_Food.json'

raw_ratings, raw_reviews, raw_metadata = load_data(rating_filepath=rating_filepath, review_filepath=review_filepath, metadata_filepath=metadata_filepath)

reviews_df, metadata_df = prepare_data(raw_ratings, raw_reviews, raw_metadata)

# Save the new dataframes to later use. 
reviews_df.to_csv('data/reviews_df.csv',index=False)
metadata_df.to_csv('data/metadata_df.csv',index=False)


