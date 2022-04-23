from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data 
df = pd.read_csv('data/metadata_df_preprocessed.csv')

# Number of products in each category
def get_category(row, categories):
    if row in categories:
        return row
    else:
        return ''

top = 20
categories = df['category'].value_counts().sort_values(ascending=False).index[0:top].to_list()
df_category = df.copy(deep=True)
df_category['category'] = df_category['category'].apply(lambda row: get_category(row, categories))
df_category = df_category[df_category['category'] != '']

# Number of ratings in each category
df_num_ratings = df[['category','num_ratings']].groupby(by=["category"]).sum(["num_ratings"])
df_num_ratings = df_num_ratings['num_ratings'].sort_values(ascending=False).reset_index()

# Plot 
fig, ((ax1, ax2)) = plt.subplots(1,2)
sns.countplot(x="category", data=df_category, order=categories, ax=ax1)
ax1.set_ylabel('Number of products')
ax1.set_xticklabels(categories,rotation=90)
sns.barplot(x="category", y="num_ratings", data=df_num_ratings[0:top], ax=ax2)
ax2.set_ylabel('Number of ratings')
ax2.set_xticklabels(df_num_ratings.loc[0:(top-1),'category'].to_list(),rotation=90)
fig.tight_layout()
fig.show()

# Variance of average ratings in each category 
categories_union = list(set().union(categories,df_num_ratings.loc[0:top,'category'])) # list of categories shown in figure 1 and 2
df_mean_avg_rating = df[df['category'].isin(categories_union)].groupby('category').median(['avg_rating']).sort_values(by='avg_rating',ascending=False)
categories_union = df_mean_avg_rating.index.to_list()
f3 = plt.figure(3)
sns.boxplot(x = 'category', y = 'avg_rating', data = df[df['category'].isin(categories_union)], order = categories_union)
plt.xticks(rotation=90)
plt.tight_layout()
f3.show()

# Select category 
category = 'Candy & Chocolate'
df_cat = df[df['category']==category]
numeric_features = ['avg_rating','std_rating','num_ratings','also_buy','price']

# Pairs plot of products in chosen category 
# f4 = plt.figure(4)
# sns.pairplot(df_cat[numeric_features])
# plt.xticks(rotation=90)
# plt.tight_layout
# f4.show()
# Observation biased data as no badly rated products have many ratings.

# Products that sales a lot but has low average rating
df_cat = df_cat.sort_values(by='rank',ascending=False)
df_cat_low_avg_rating = df_cat[df_cat['avg_rating'] < 3]
df_cat_low_avg_rating = df_cat_low_avg_rating[df_cat_low_avg_rating['num_ratings'] >= 200]
df_cat_low_avg_rating.shape
df_cat_low_avg_rating[['category','avg_rating','num_ratings','rank','price']]

# Save dataframes for 3 categories
category = 'Beverages'
df_beverages = df[df['category']==category]
df_beverages.to_csv('data/df_beverages.csv')

category = 'Candy & Chocolate'
df_candy_chocolate = df[df['category']==category]
df_candy_chocolate.to_csv('data/df_candy_chocolate.csv')

category = 'Snack Foods'
df_snacks = df[df['category']==category]
df_snacks.to_csv('data/df_snacks.csv')

