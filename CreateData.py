### Load packages
import pandas as pd

### Import raw data
def load_data(rating_filepath, review_filepath):
    ratings_df = pd.read_csv(rating_filepath, names = ['item','user','rating','timestamp'])
    reviews_df = pd.read_json(review_filepath, lines=True)

    return ratings_df, reviews_df

