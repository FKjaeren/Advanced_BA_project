### Load packages
import pandas as pd

### Import raw data
def load_data(rating_filepath, review_filepath, metadata_filepath):
    ratings_df = pd.read_csv(rating_filepath, names = ['item','user','rating','timestamp'])
    reviews_df = pd.read_json(review_filepath, lines=True)
    metadata_df = pd.read_json(metadata_filepath, lines=True)
    return ratings_df, reviews_df, metadata_df

