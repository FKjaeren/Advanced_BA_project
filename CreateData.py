### Load packages
import pandas as pd

### Import raw data
def load_data(rating_filepath, review_filepath, something, metadata_filepath):
    ratings_df = pd.read_csv(rating_filepath, names = ['item','user','rating','timestamp'])
    reviews_df = pd.read_json(review_filepath, lines=True)
    metadata_df = pd.read_json(metadata_filepath, lines=True)
    something_df = pd.read_json(something, lines=True)
    return ratings_df, reviews_df, something_df, metadata_df

