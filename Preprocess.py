# Import packages
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gower
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.feature_extraction import _stop_words
import string
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Function to preprocess the train and test dataset
def preprocess_data(df_train, df_test):
    # get number of also_buy
    df_train['also_buy'] = df_train['also_buy'].fillna('').apply(get_number_also_buy)
    df_test['also_buy'] = df_test['also_buy'].fillna('').apply(get_number_also_buy)

    # sales rank information
    df_train['rank'] = df_train['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_train['rank'] = pd.to_numeric(df_train['rank'], errors = 'coerce').fillna(0).apply(int)
    df_test['rank'] = df_test['rank'].apply(get_rank).str.replace(',','').str.extract('(\d+|$)')
    df_test['rank'] = pd.to_numeric(df_test['rank'], errors = 'coerce').fillna(0).apply(int)

    # Fill nan values for price data
    # get number of also_view
    df_train['also_view'] = df_train['also_view'].fillna('').apply(get_number_also_buy)
    df_test['also_view'] = df_test['also_view'].fillna('').apply(get_number_also_buy)

    # Clean description
    df_train['description'] = df_train['description'].apply(get_description)
    # Drop rows with no information
    df_train = df_train.dropna(axis = 0, subset=['description'])
    # Make it a string and clean html text
    df_train['description'] = df_train['description'].apply(str)
    df_train['description'] = df_train['description'].str.replace('\n', '')
    df_train['description'] = df_train[['description']].applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    # Perform text processing where stop words are removed etc. 
    df_train['description'] = df_train['description'].apply(text_processing)

    # Do the same for test dataset
    df_test['description'] = df_test['description'].apply(get_description)
    df_test = df_test.dropna(axis = 0, subset=['description'])
    df_test['description'] = df_test['description'].apply(str)
    df_test['description'] = df_test['description'].str.replace('\n', '')
    df_test['description'] = df_test[['description']].applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    df_test['description'] = df_test['description'].apply(text_processing)
    return df_train, df_test

# Number of also bougtht products
def get_number_also_buy(row):
    number = len(row)
    return number
# Get the brand
def get_brand(row, brands):
    if row in brands:
        return row
    else:
        return 'Other'
# Get the rank from list
def get_rank(row):
    if isinstance(row, list):
        if len(row) > 0:
            return row[0]
        else:
            return ''
    else:
        return row

# Use only rows with list of information else nan
def get_description(row):
    if isinstance(row, list):
        if len(row)>0:
            return row
        else:
            return np.nan
    else:
        return row

# Function used to clean text data
def text_processing(text):
    # remove punctuation 
    text = "".join([c for c in text 
        if c not in string.punctuation])
    # lowercase
    text = "".join([c.lower() for c in text])
    # stemming / lematizing (optional)
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    # remove stopwords
    text = " ".join([w for w in text.split() 
        if w not in Stop_Words])
    return text
# Function for preprocessing the prize where the data also will splitted in a train and a test set. 
def preprocess_price(metadata_df):
    # Drop columns we don't use
    df = metadata_df.drop(columns = ['item','title','feature','main_cat','similar_item','details','timestamp'])
    # Empty columns can't be used for anything
    df = df.dropna(axis=0,subset=['avg_rating','num_ratings','description'])
    # Split the data in a 75 % split train and test set. 
    df_train, df_test = train_test_split(df, train_size=0.75)

    # Below we find the mean price for every cateogry. 
    categories = []
    category_means = []
    categories = df_train.category.unique()
    for i in categories:
        temp = df_train[df_train['price'].isna() == False]
        mean_value = temp[temp['category'] == i]['price'].mean()
        category_means.append(mean_value)
    dict = {'categories': categories,'category_means': category_means}
    category_stat_df = pd.DataFrame(dict)
    category_stat_df = category_stat_df.set_index('categories')

    # Next for all NULL values we assign the price to be the mean price in that category.

    df_train['price'] = df_train.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)
    df_test['price'] = df_test.apply(lambda row: category_stat_df.loc[row['category']].values[0] if row['price'] != row['price'] else row['price'], axis = 1)
    
    # Here we drop the category column. 
    columns = df_train.columns
    if 'category' in columns:
        df_train = df_train.drop(columns = ['category'])
        df_test = df_test.drop(columns = ['category'])
    if 'orig category' in columns:
        df_train = df_train.drop(columns = ['orig category'])
        df_test = df_test.drop(columns = ['orig category'])
    return df_train, df_test



# Words from already trained lda model. The words are removed because they either occur in multiple topics or doesn't make sense in the topic. 
category = 'Candy & Chocolate'
if category == 'Candy & Chocolate':
    Stop_Words = _stop_words.ENGLISH_STOP_WORDS.union(['chocolate','supplement','cocoa','candy','cure','condition','flavor','statement','product',
                                                        'health','milk','intended','dietary', 'prevent','fda','diagnose','regarding',
                                                        'evaluated','disease','treat','sugar','sweet','oz','40','color'])
elif category == 'Snack Foods':
    Stop_Words = _stop_words.ENGLISH_STOP_WORDS.union(['snack','food','fda','flavor','product','ingredient','statement'])
elif category == 'Beverages':
    Stop_Words = _stop_words.ENGLISH_STOP_WORDS.union(['tea','coffee','water','cup','supplement','flavor','year','food','condition'])

# First we read the "prepared" data and the proper category found in the "exploratory part"
metadata_df = pd.read_csv('data/'+category+'/df_'+category+'.csv')

#We clone the category, as we will need this in order to preprocess price, but we will remove the original category column, 
# when we create dummy data
metadata_df['orig category'] = metadata_df['category']
dummy_df = pd.get_dummies(metadata_df, columns=['brand','orig category'])
# We drop brand as we can't have non numerical values. 
metadata_df = metadata_df.drop(columns=['brand'])

# We apply the preprocess price function, which handles some of the preprocess functions as well as splits the data in train and test.
df_train, df_test = preprocess_price(metadata_df)
df_train_dummy, df_test_dummy = preprocess_price(dummy_df)

# Next the second part of the preprocess will be applied.
df_train, df_test = preprocess_data(df_train, df_test)
df_train_dummy, df_test_dummy = preprocess_data(df_train_dummy, df_test_dummy)

## We drop description when we do pca on the data as it is non-numerical. Also we drop std_rating as it is quite hightly correlated with 
# Avg. rating, and it seems unlikely that std_rating can be gathered in a situation where average rating (the target) can't.
df_train_dummy = df_train_dummy.drop(columns = ['description','std_rating'])
df_test_dummy = df_test_dummy.drop(columns = ['description','std_rating'])

## Next we decompose our data.
pca = PCA(n_components=100)
pca.fit(df_train_dummy)
print(pca.explained_variance_ratio_)

## The explained variance have been calculated, and it seems only 2 components are neccesary.
pca = PCA(n_components=3).fit(df_train_dummy)
pca_values = pca.fit_transform(df_train_dummy)

### Plot the pca values in order to identify if there are any clusters:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_values[:,0], pca_values[:,1], pca_values[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

plt.scatter(pca_values[:,0], pca_values[:,1])
plt.show()

## Then we create 5 clusters using KMeans.
kmeans = KMeans(n_clusters=2).fit(pca_values)

## We assign these clusters to our original (preprocessed) data.
df_train['cluster'] = kmeans.labels_
pca_values_test = pca.transform(df_test_dummy)
df_test['cluster'] = kmeans.predict(pca_values_test)

## We one hot encode the clusters, so that any model trained on the data, wan't assume there is a relationship, between the values.
df_train = pd.get_dummies(df_train, columns = ['cluster'])
df_test = pd.get_dummies(df_test, columns = ['cluster'])

# ensure same shape of train and test
if df_train.shape[1] != df_test.shape[1]:
    setdiff = set(df_train.columns).difference(set(df_test.columns))
    for name in setdiff:
        df_test[name] = np.zeros(df_test.shape[0])
        df_test = df_test.astype({name:'int'})
        
# order columns in test set
df_test = df_test[df_train.columns]     

#sns.scatterplot('price','num_ratings', hue = 'cluster', data = df_train)
#plt.show()

df_train.to_csv('data/' + category + '/df_train.csv',index=False)
df_test.to_csv('data/' + category + '/df_test.csv',index=False)
