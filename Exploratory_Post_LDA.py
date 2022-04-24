import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_candy_chocolate = pd.read_csv('data/df_Candy & Chocolate_with_lda.csv')

df_candy_chocolate = df_candy_chocolate.drop(columns=['category'])

sns.heatmap(df_candy_chocolate.corr().round(2), cmap='Blues', annot=True)\
   .set_title('Correlation matrix')
plt.show()

df_candy_chocolate['top_brand'].hist()
plt.show()

top_brands=["Jelly Belly", "Black Tie Mercantile", "Nestle","The Nutty Fruit House","Trader Joe's"]
test = df_candy_chocolate[df_candy_chocolate['top_brand'] in top_brands]
