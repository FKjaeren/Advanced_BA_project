# Import packages
from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Investigate categories 

# lead data 
df = pd.read_csv('data/metadata_df.csv')

# Number of products in each category
def get_category(row, categories):
    if row in categories:
        return row
    else:
        return ''

# We will only look at the top 20 categories
top = 20
categories = df['category'].value_counts().sort_values(ascending=False).index[0:top].to_list()

# Copy the dataframe
df_category = df.copy(deep=True)

# Return number of products in each category
df_category['category'] = df_category['category'].apply(lambda row: get_category(row, categories))
df_category = df_category[df_category['category'] != '']

# Number of ratings in each category
df_num_ratings = df[['category','num_ratings']].groupby(by=["category"]).sum(["num_ratings"])
df_num_ratings = df_num_ratings['num_ratings'].sort_values(ascending=False).reset_index()

# Plot of the number of productts in each category and the number of ratings in each category
fig, ((ax1, ax2)) = plt.subplots(1,2)
sns.countplot(x="category", data=df_category, order=categories, ax=ax1)
ax1.set_ylabel('Number of products')
ax1.set_xticklabels(categories,rotation=90)
sns.barplot(x="category", y="num_ratings", data=df_num_ratings[0:top], ax=ax2)
ax2.set_ylabel('Number of ratings')
ax2.set_xticklabels(df_num_ratings.loc[0:(top-1),'category'].to_list(),rotation=90)
fig.tight_layout()
fig.show()

# Plot with variance of average ratings in each category 
categories_union = list(set().union(categories,df_num_ratings.loc[0:top,'category'])) # list of categories shown in figure 1 and 2
df_mean_avg_rating = df[df['category'].isin(categories_union)].groupby('category').median(['avg_rating']).sort_values(by='avg_rating',ascending=False)
categories_union = df_mean_avg_rating.index.to_list()
plt.figure(2)
sns.boxplot(x = 'category', y = 'avg_rating', data = df[df['category'].isin(categories_union)], order = categories_union)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Select category 
category = 'Snack Foods'
df_cat = df[df['category']==category]

# Save dataframe for category 
df_cat.to_csv('data/'+category+'/df_'+category+'.csv',index=False)
