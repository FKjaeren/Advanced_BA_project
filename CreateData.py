### Load packages
import pandas as pd

### Import raw data
def load_data(rating_filepath, review_filepath):
    ratings_df = pd.read_csv(rating_filepath, names = ['item','user','rating','timestamp'])
    reviews_df = pd.read_json(review_filepath, lines=True)

    return ratings_df, reviews_df

rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json'

ratings_df, reviews_df = load_data(rating_filepath, review_filepath)

### Show raw data
ratings_df.head()
reviews_df.head()

reviews_df.columns

ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], origin = 'unix', unit = 's')

### Time span = 2000-06-19 to 2018-10-07
reviews_df['timestamp'] = pd.to_datetime(reviews_df['unixReviewTime'], origin = 'unix', unit = 's')