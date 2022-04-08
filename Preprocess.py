from CreateData import load_data
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction import _stop_words
import string
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
Stop_Words= _stop_words.ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup



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

    # group ratings_df and merge with metadata
    grouped_ratings = ratings_df[['item','rating']].groupby(by='item').agg({'rating':'mean','item':'size'}).rename(columns={'rating':'avg_rating','item':'num_ratings'}).reset_index()
    metadata_df = grouped_ratings.merge(metadata_df, how='outer', left_on='item', right_on='asin')
    metadata_df['item'].fillna(metadata_df['asin'], inplace=True)
    metadata_df = metadata_df.drop(columns=['asin','date','tech1','tech2','fit'])

    # preprocess price
    metadata_df['price'] =  pd.to_numeric(metadata_df['price'].str.replace('$',''), errors='coerce')

    return reviews_df, metadata_df

def preprocess_data(metadata_df):
    
    X = metadata_df[['avg_rating','num_ratings', 'category', 'also_buy', 'brand', 'rank',
       'also_view', 'price','description']]

    # get category
    X['category'] = X['category'].fillna('')
    X['category'] = X['category'].apply(get_category)

    # get number of also_buy
    X['also_buy'] = X['also_buy'].fillna('')
    X['also_buy'] = X['also_buy'].apply(get_number_also_buy)

    # get top 10 brands
    top = 10
    X['brand'] = X['brand'].str.replace('Unknown','')
    brands = X['brand'].value_counts().sort_values(ascending=False).index[1:(top+1)].to_list()
    X['top_brand'] = X['brand'].apply(lambda row: get_brand(row, brands))
    X = X.drop(columns=['brand'])

    # sales rank information
    X['rank'] = X['rank'].apply(get_rank)
    X['rank'] = X['rank'].str.replace(',','')
    X['rank'] = X['rank'].str.extract('(\d+|$)')
    X['rank'] = pd.to_numeric(X['rank'], errors = 'coerce').fillna(0).apply(int)

    # Fill nan values for price data
    categories = []
    category_means = []
    categories = X.category.unique()
    for i in categories:
        temp = X[X['price'].isna()==False]
        mean_value = temp[temp['category']==i]['price'].mean()
        category_means.append(mean_value)

    dict = {'categories': categories,'category_means':category_means}

    category_stat_df = pd.DataFrame(dict)
    category_stat_df = category_stat_df.set_index('categories')

    X['price'] = X.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)

    # get dummies for: category, top_brands
    # get number of also_view
    X['also_view'] = X['also_view'].fillna('')
    X['also_view'] = X['also_view'].apply(get_number_also_buy)

    X['description'] = X['description'].apply(str)
    X['description'] = X['description'].str.replace('\n', '')

    X['description'] = X[['description']].applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    X['description'] = X['description'].apply(text_processing)

    # drop nan's
    X = X.dropna(axis=0,subset=['avg_rating','num_ratings','category'])

    # get dummies for: category, top_brand
    X = pd.get_dummies(X, columns=['category','top_brand'])

    #y = X['avg_rating']
    #X = X.drop(columns=['avg_rating'])
    return X

def get_category(row):
    if len(row) > 1:
        category = row[1]
    else:
        category = row
    return category

def get_number_also_buy(row):
    number = len(row)
    return number

def get_brand(row, brands):
    if row in brands:
        return row
    else:
        return ''

def get_rank(row):
    if isinstance(row, list):
        if len(row) > 0:
            return row[0]
        else:
            return ''
    else:
        return row

def text_processing(text):
    # remove punctuation 
    text = "".join([c for c in text 
                    if c not in string.punctuation])
    # lowercase
    text = "".join([c.lower() for c in text])
    # remove stopwords
    text = " ".join([w for w in text.split() 
                     if w not in Stop_Words])
    # stemming / lematizing (optional)
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

rating_filepath = 'data/Grocery_and_Gourmet_Food.csv'
review_filepath = 'data/Grocery_and_Gourmet_Food_5.json' # Har ændret til gz da min fil hedder det, ved ikke hvad forskel det gør for jer
metadata_filepath = 'data/meta_Grocery_and_Gourmet_Food.json'

raw_ratings, raw_reviews, raw_metadata = load_data(rating_filepath=rating_filepath, review_filepath=review_filepath, metadata_filepath=metadata_filepath)
#Counter(" ".join(raw_metadata["description_clean"]).split()).most_common(1000)

reviews_df, metadata_df = prepare_data(raw_ratings, raw_reviews, raw_metadata)
metadata_df_clean = preprocess_data(metadata_df)

#X = X.drop(columns='price')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#reg = LinearRegression().fit(X_train, y_train)
#print(reg.score(X_test, y_test))

reviews_df.to_csv('data/reviews_df.csv',index=False)
metadata_df_clean.to_csv('data/metadata_df.csv',index=False)

