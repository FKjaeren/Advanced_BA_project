import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pickle
from datetime import date

# set random seed
set.seed(42)

df = pd.read_csv('data/metadata_df_preprocessed.csv')

def get_len(text):
    if text != text:
        return 0
    elif isinstance(text, float):
        return 1
    else:
        return len(text)

def train_lda(df, n_topics, text):
    df['message_len'] = df[text].apply(get_len)
    count_vect = CountVectorizer()
    bow_counts = count_vect.fit_transform(df.dropna(subset=[text])[text].values)
    print('Vocabulary size = ',len(count_vect.vocabulary_))

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.)

    X_len = df['message_len'].values
    print(X_len)
    X_len = X_len.reshape(-1, 1) # Since we it is single feature
    X_bow_counts = bow_counts
    #(X_len_train, X_len_test, X_bow_counts_train, X_bow_counts_test) = train_test_split(X_len[X_len!=0], X_bow_counts, test_size=0.2)

    X_lda = lda.fit_transform(X_bow_counts)
    return X_lda, lda, count_vect

n_topics = 10
X_lda, lda, count_vect = train_lda(df = df, n_topics=n_topics, text = 'description')

def print_top_words(model, feature_names, n_top_words):
    norm = model.components_.sum(axis=1)[:, np.newaxis]
    for topic_idx, topic in enumerate(model.components_):
        print(80 * "-")
        print("Topic {}".format(topic_idx))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            print("{:.3f}".format(topic[i] / norm[topic_idx][0]) 
                  + '\t' + feature_names[i])

print("\nTopics in LDA model:")
counts_feature_names = count_vect.get_feature_names()
n_top_words = 10
print_top_words(lda, counts_feature_names, n_top_words)
lda_list = []
for i in range(n_topics):
    lda_list.append('lda'+str(i+1))
X_lda_df = pd.DataFrame(X_lda, columns = lda_list)

X_lda_df.to_csv('data/lda_data_df.csv',index=False)
today = date.today()

filename = 'models/lda_model'+str(today)+'.sav'
pickle.dump(lda, open(filename, 'wb'))

filename = 'models/count_vect_model'+str(today)+'.sav'
pickle.dump(count_vect, open(filename, 'wb'))
