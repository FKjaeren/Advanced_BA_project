from CreateData import load_data
import seaborn as sns
import pandas as pd

def preprocess_data(ratings_df, reviews_df):
    # create timestamps
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], origin = 'unix', unit = 's')
    reviews_df['timestamp'] = pd.to_datetime(reviews_df['unixReviewTime'], origin = 'unix', unit = 's')

    # drop columns in reviews
    reviews_df = reviews_df.drop(columns=['unixReviewTime','reviewTime','reviewerName','vote','image','style','verified'])
    
    # drop na's and duplicates
    reviews_df = reviews_df.dropna()
    reviews_df = reviews_df.drop_duplicates(keep='first')
    ratings_df = ratings_df.drop_duplicates(keep='first')

    return ratings_df, reviews_df

rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json'

raw_ratings, raw_reviews = load_data(rating_filepath, review_filepath)

# rating ['item', 'user', 'rating', 'timestamp']
# review ['overall','reviewerID','asin','reviewText','summary','timestamp]
ratings_df, reviews_df = preprocess_data(raw_ratings, raw_reviews)




