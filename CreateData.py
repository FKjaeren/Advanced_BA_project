### Load packages
import pandas as pd

### Import raw data
rating_df = pd.read_csv('data/Grocery_and_Gourmet_Food.csv', names = ['item','user','rating','timestamp'])
reviews_df = pd.read_json('data/Grocery_and_Gourmet_Food_5.json', lines=True)

### Show raw data
rating_df.head()
reviews_df.head()

###
reviews_df.columns

### 
rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], origin = 'unix', unit = 's')

### Time span = 2000-06-19 to 2018-10-07