import pandas as pd

metadata_df = pd.read_csv('data/metadata_df_clean.csv')

y = metadata_df['avg_rating']
X = metadata_df.drop(columns=['avg_rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
