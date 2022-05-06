## Load packages
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

## Define category to test
category = 'Candy & Chocolate'

## Load models
filename = 'models/'+category+'/lda_model_2022-05-03.sav'
lda_model = pickle.load(open(filename, 'rb'))

filename = 'models/'+category+'/count_vect_model_2022-05-03.sav'
count_vect_model = pickle.load(open(filename, 'rb'))

## Define function to print top N words in every topic of a LDA model.
def print_top_words(model, feature_names, n_top_words):
    norm = model.components_.sum(axis=1)[:, np.newaxis]
    for topic_idx, topic in enumerate(model.components_):
        print(80 * "-")
        print("Topic {}".format(topic_idx))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            print("{:.3f}".format(topic[i] / norm[topic_idx][0]) 
                  + '\t' + feature_names[i])

## Extract feature names
counts_feature_names = count_vect_model.get_feature_names()
# Define N top words to print.
n_top_words = 10

# Print the top N words for each topic.
print_top_words(lda_model, counts_feature_names, n_top_words)

## Load the best performing model to predict out target variable.
filename = 'models/'+category+'/tuned_gb_regressor_2022-05-03.sav'
best_model = pickle.load(open(filename, 'rb'))

## Read test data
test_df = pd.read_csv('data/'+category+'/df_test_lda.csv')
test_df = test_df.drop(columns = ['description','std_rating'])

# Define the X values and y values.
X_test = test_df.drop(columns=['avg_rating'])
y_test = test_df['avg_rating']

# Predict the target variable using the X values.
y_gb_regressor = best_model.predict(X_test)

## Calculate the MAE for this testset.
MAE_gb_regressor = mean_absolute_error(y_test, y_gb_regressor)

print("MAE = ",MAE_gb_regressor)