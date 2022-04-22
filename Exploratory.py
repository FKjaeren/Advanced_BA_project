import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/metadata_df.csv')

sns.pairplot(df)
plt.show()
# Observation biased data as no badly rated products have many ratings.

sns.displot(x="avg_rating", data=df)
plt.show()


metadata_cleaned_df = pd.read_csv('data/metadata_df.csv')

lda_transformed_df = pd.read_csv('data/lda_data_df.csv')