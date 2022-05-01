from math import prod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

category = 'Candy & Chocolate'
df_train = pd.read_csv('data/'+category+'/df_train_lda.csv')

df_train = df_train.drop(columns=['description'])

sns.heatmap(df_train.corr().round(2), cmap='Blues', annot=True)\
   .set_title('Correlation matrix')
plt.show()

df_subset = df_train[df_train['price']<80]
df_subset = df_subset[df_subset['also_buy']<400]
#df_subset = df_subset[df_subset['rank']]
df_subset = df_subset[['avg_rating', 'std_rating', 'num_ratings', 'description', 'also_buy',
       'rank', 'also_view', 'price','lda1',
       'lda2', 'lda3', 'lda4', 'lda5']]
df_subset[df_subset['num_ratings']<50].hist(bins = 20)
plt.show()


sns.heatmap(df_subset.corr().round(2), cmap='Blues', annot=True)\
   .set_title('Correlation matrix')
plt.show()

### find the right product to enchance

## We find subsets of the data, to limit the amount of products to check.

## Finding the 75% quantile of number of ratings
df_train['num_ratings'].describe()

## Defining that subset
products_to_enhance = df_train[df_train['num_ratings']>df_train['num_ratings'].describe().loc['75%']]

products_to_enhance['num_ratings'].describe()

## Finding the 75% quantile of the 75% quantile of the number of ratings
products_to_enhance = products_to_enhance[products_to_enhance['num_ratings']>=products_to_enhance['num_ratings'].describe().loc['75%']]


products_to_enhance['avg_rating'].describe()

## Finding the 25% quantile of avg. ratings of the subset.

products_to_enhance = products_to_enhance[products_to_enhance['avg_rating'] <= df_train['avg_rating'].describe().loc['25%'] ]

products_to_enhance['num_ratings'].describe()

## finding the 75% quantile of number of ratings from the subset
products_to_enhance = products_to_enhance[products_to_enhance['num_ratings']>=products_to_enhance['num_ratings'].describe().loc['75%']]

products_to_enhance['std_rating'].describe()

## Finding the 25% quantile of standard deviations for ratings in the subset
products_to_enhance = products_to_enhance[products_to_enhance['std_rating'] <= products_to_enhance['std_rating'].describe().loc['25%'] ]

#products_to_enhance = products_to_enhance[products_to_enhance['rank'] <= products_to_enhance['rank'].describe().loc['75%'] ]

product_to_enhance = products_to_enhance.loc[products_to_enhance['avg_rating'] == min(products_to_enhance['avg_rating'])]

sns.heatmap(products_to_enhance.corr().round(2), cmap='Blues', annot=True)\
   .set_title('Correlation matrix')
plt.show()

