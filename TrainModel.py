import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

metadata_df = pd.read_csv('data/metadata_df_clean.csv')

y = metadata_df['avg_rating']
X = metadata_df.drop(columns=['avg_rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))