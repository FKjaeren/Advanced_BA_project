## Load packages
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
import wordcloud
import matplotlib.pyplot as plt

## Define category to test
category = 'Candy & Chocolate'

## Load models
filename = 'models/'+category+'/lda_model_4_components2022-05-08.sav'
lda_model = pickle.load(open(filename, 'rb'))

filename = 'models/'+category+'/count_vect_model_2022-05-08.sav'
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

def visualize_topics(lda, count_vect, terms_count):
    terms = count_vect.get_feature_names()
    for idx, topic in enumerate(lda.components_):
        title = 'Topic ' + str(idx+1)
        abs_topic = abs(topic)
        topic_terms = [[terms[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
        topic_terms_sorted = [[terms[i], topic[i]] for i in abs_topic.argsort()[:-terms_count - 1:-1]]
        topic_words = []
        for i in range(terms_count):
            topic_words.append(topic_terms_sorted[i][0])
            # print(','.join( word for word in topic_words))
            # print("")
            dict_word_frequency = {}
        for i in range(terms_count):
            dict_word_frequency[topic_terms_sorted[i][0]] = topic_terms_sorted[i][1]
            wcloud = wordcloud.WordCloud(background_color="white",mask=None, max_words=100,\
            max_font_size=60,min_font_size=10,prefer_horizontal=0.9,
            contour_width=3,contour_color='black')
            wcloud.generate_from_frequencies(dict_word_frequency)
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=20)
        # plt.savefig("Topic#"+str(idx+1), format="png")
        plt.show()
    return

## Extract feature names
counts_feature_names = count_vect_model.get_feature_names()
# Define N top words to print.
n_top_words = 10

# Print the top N words for each topic.
print_top_words(lda_model, counts_feature_names, n_top_words)
visualize_topics(lda_model, count_vect_model, 10)

## Load the best performing model to predict out target variable.
filename = 'models/'+category+'/best_performing_model_4_2022-05-08.sav'
best_model = pickle.load(open(filename, 'rb'))

## Read test data
test_df = pd.read_csv('data/'+category+'/df_test_4_components_lda.csv')
test_df = test_df.drop(columns = ['description','std_rating','item'])

# Define the X values and y values.
X_test = test_df.drop(columns=['avg_rating'])
y_test = test_df['avg_rating']

# Predict the target variable using the X values.
y_gb_regressor = best_model.predict(X_test)

## Calculate the MAE for this testset.
MAE_gb_regressor = mean_absolute_error(y_test, y_gb_regressor)

print("MAE = ",MAE_gb_regressor)
print(best_model)