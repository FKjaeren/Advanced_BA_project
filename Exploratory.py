import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/metadata_df.csv')
df = df.drop(columns=['Unnamed: 0'])

sns.pairplot(df)
plt.show()
# Observation biased data as no badly rated products have many ratings.

sns.displot(x="avg_rating", data=df)
plt.show()