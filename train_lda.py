# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from datetime import date
import wordcloud
import matplotlib.pyplot as plt
from nltk import word_tokenize
# set random seed
np.random.seed(42)

#%% Functions 
def get_len(text):
    if text != text:
        return 0
    elif isinstance(text, float):
        return 1
    else:
        return len(text)

# Function where LDA is trained on both the trainf and test. Both datasets are returned along with the model and count_vect
def train_lda(df_train, df_test, gridsearch, text):
    docs_train = df_train[text].values
    #docs_train = np.vectorize(docs_train)
    #docs_train = [word_tokenize(d) for d in docs_train]
    docs_test = df_test[text]
    #docs_test = [word_tokenize(d) for d in docs_test]
    count_vect = CountVectorizer()
    bow_counts_train = count_vect.fit_transform(docs_train)
    bow_counts_test = count_vect.transform(docs_test)

    cv_matrix = count_vect.fit_transform(docs_train)
    gridsearch.fit(cv_matrix)

    ## Save the best model
    best_lda = gridsearch.best_estimator_
    # What did we find?
    print("Best Model's Params: ", gridsearch.best_params_)

    # Train LDA with best params
    n_topics = gridsearch.best_params_['n_components']
    ld = gridsearch.best_params_['learning_decay']

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                learning_method='batch',
                                learning_offset=10.,
                                learning_decay=ld)

    X_train_lda = lda.fit_transform(bow_counts_train)
    X_test_lda = lda.transform(bow_counts_test)

    return X_train_lda, X_test_lda, lda, count_vect, best_lda, n_topics


# Function to print the top words in a topic
def print_top_words(model, feature_names, n_top_words):
    norm = model.components_.sum(axis=1)[:, np.newaxis]
    for topic_idx, topic in enumerate(model.components_):
        print(80 * "-")
        print("Topic {}".format(topic_idx))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            print("{:.3f}".format(topic[i] / norm[topic_idx][0]) 
                  + '\t' + feature_names[i])

# Function to visualize the topics in a wordcloud
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

#%% RUN LDA

# Get data
category = 'Candy & Chocolate'
train_path = 'data/' + category + '/df_train.csv'
test_path = 'data/' + category + '/df_test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_train = df_train.dropna(axis=0,subset=['description'])
df_test = df_test.dropna(axis=0,subset=['description'])

# Options to tune hyperparamets in LDA model
# Beware it will try *all* of the combinations, so it'll take ages
search_params = {'n_components': [4, 6, 8], 'learning_decay': [0.5, .7]}

# Set up LDA with the options we'll keep static
model = LatentDirichletAllocation(learning_method='online',
                                  max_iter=5,
                                  random_state=0)

# Try all of the options
gridsearch = GridSearchCV(model,
                          param_grid=search_params,
                          n_jobs=-1,
                          verbose=2,
                         )
count_vect = CountVectorizer()


# Run LDA on description with tuned parameters
X_train_lda, X_test_lda, lda, count_vect, best_lda, n_topics = train_lda(df_train, df_test, gridsearch, 'description')

# Visualize topics as wordclouds
visualize_topics(lda, count_vect, 25)

# Merge df with lda 
lda_list = []
for i in range(n_topics):
    lda_list.append('lda'+str(i+1))
X_train_lda_df = pd.DataFrame(X_train_lda, columns = lda_list)
X_test_lda_df = pd.DataFrame(X_test_lda, columns = lda_list)
df_train_lda = df_train.merge(X_train_lda_df, left_index=True, right_index=True)
df_test_lda = df_test.merge(X_test_lda_df, left_index=True, right_index=True)

# Save merged data + model
today = date.today()
df_train_lda.to_csv('data/' + category + '/df_train_4_components_lda.csv',index=False)
df_test_lda.to_csv('data/' + category + '/df_test_4_components_lda.csv',index=False)
filename = 'models/'+category+'/lda_model_4_components'+str(today)+'.sav'
pickle.dump(lda, open(filename, 'wb'))
filename = 'models/'+category+'/count_vect_model_'+str(today)+'.sav'
pickle.dump(count_vect, open(filename, 'wb'))

# %%
