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

def train_lda(df, n_topics, ld, text):
    df['message_len'] = df[text].apply(get_len)
    count_vect = CountVectorizer()
    #bow_counts = count_vect.fit_transform(df.dropna(subset=[text])[text].values)
    bow_counts = count_vect.fit_transform(df[text].values)
    print('Vocabulary size = ',len(count_vect.vocabulary_))

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                learning_decay=ld)

    X_len = df['message_len'].values
    print(X_len)
    X_len = X_len.reshape(-1, 1) # Since we it is single feature
    X_bow_counts = bow_counts
    #(X_len_train, X_len_test, X_bow_counts_train, X_bow_counts_test) = train_test_split(X_len[X_len!=0], X_bow_counts, test_size=0.2)

    X_lda = lda.fit_transform(X_bow_counts)
    return X_lda, lda, count_vect

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

#%% RUN LDA

# get data
category = 'Candy & Chocolate' # 'all' if you want to select data with all categories 
if category != all:
    path = 'data/df'+'_'+category+'.csv'
else:
    path = 'data/metadata_df_preprocessed.csv'
df = pd.read_csv(path)

# Options to try with our LDA
# Beware it will try *all* of the combinations, so it'll take ages
search_params = {'n_components': [1, 5], 'learning_decay': [.5, .7]}

# Set up LDA with the options we'll keep static
model = LatentDirichletAllocation(learning_method='online',
                                  max_iter=5,
                                  random_state=0)

# Try all of the options
gridsearch = GridSearchCV(model,
                          param_grid=search_params,
                          n_jobs=-1,
                          verbose=1)
count_vect = CountVectorizer()
cv_matrix = count_vect.fit_transform(df['description'].values)
gridsearch.fit(cv_matrix)

## Save the best model
best_lda = gridsearch.best_estimator_

# What did we find?
print("Best Model's Params: ", gridsearch.best_params_)

# Train LDA with best params
n_topics = gridsearch.best_params_['n_components']
learning_decay = gridsearch.best_params_['learning_decay']
X_lda, lda, count_vect = train_lda(df, n_topics, learning_decay, 'description')

# Visualize topics as wordclouds
visualize_topics(lda, count_vect, 25)

# print top words from lda model 
print("\nTopics in LDA model:")
counts_feature_names = count_vect.get_feature_names()
n_top_words = 10
print_top_words(lda, counts_feature_names, n_top_words)
lda_list = []
for i in range(n_topics):
    lda_list.append('lda'+str(i+1))
X_lda_df = pd.DataFrame(X_lda, columns = lda_list)

# Merge df with lda 
df_with_lda = df.merge(X_lda_df, left_index=True, right_index=True)

# Save merged data + model
today = date.today()
df_with_lda.to_csv('data/df_'+category+'_with_lda.csv',index=False)
filename = 'models/lda_model_'+category+'_'+str(today)+'.sav'
pickle.dump(lda, open(filename, 'wb'))
filename = 'models/count_vect_model'+category+'_'+str(today)+'.sav'
pickle.dump(count_vect, open(filename, 'wb'))

