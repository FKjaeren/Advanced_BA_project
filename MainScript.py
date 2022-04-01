from CreateData import load_data
import pandas as pd

rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json'

ratings_df, reviews_df = load_data(rating_filepath, review_filepath)

### Show raw data
ratings_df.head()
reviews_df.head()
ratings_df.columns 
#['item', 'user', 'rating', 'timestamp']
reviews_df.columns 
#['overall','verified','reviewTime','reviewerID','asin','reviewerName','reviewText','summary','unixReviewTime','vote','style','image']

ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], origin = 'unix', unit = 's')

### Time span = 2000-06-19 to 2018-10-07
reviews_df['timestamp'] = pd.to_datetime(reviews_df['unixReviewTime'], origin = 'unix', unit = 's')