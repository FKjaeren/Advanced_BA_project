import pandas as pd

rating_df = pd.read_csv('data/Grocery_and_Gourmet_Food.csv')
reviews_df = pd.read_json('data/Grocery_and_Gourmet_Food_5.json', lines=True)

reviews_df.head()