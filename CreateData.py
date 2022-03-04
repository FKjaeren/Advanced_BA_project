### Load packages
import pandas as pd

### Import raw data
rating_df = pd.read_csv('data/Grocery_and_Gourmet_Food.csv')
reviews_df = pd.read_json('data/Grocery_and_Gourmet_Food_5.json', lines=True)

### Show raw data
rating_df.head()
reviews_df.head()