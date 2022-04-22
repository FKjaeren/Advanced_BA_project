from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

plt.figure()
sns.countplot(x="category", data=df_category, order=categories)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Number of ratings in each category
df_num_ratings = df[['category','num_ratings']].groupby(by=["category"]).sum(["num_ratings"])
df_num_ratings = df_num_ratings['num_ratings'].sort_values(ascending=False).reset_index()
top = 20

plt.figure()
sns.barplot(x="category", y="num_ratings", data=df_num_ratings[0:top])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()






sns.pairplot(df)
plt.show()
# Observation biased data as no badly rated products have many ratings.

sns.displot(x="avg_rating", data=df)
plt.show()


metadata_cleaned_df = pd.read_csv('data/metadata_df.csv')

lda_transformed_df = pd.read_csv('data/lda_data_df.csv')