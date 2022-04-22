import pickle
import numpy as np
import pandas as pd

filename = 'models/lda_model2022-04-22.sav'
lda_model = pickle.load(open(filename, 'rb'))

filename = 'models/count_vect_model2022-04-22.sav'
count_vect_model = pickle.load(open(filename, 'rb'))

def print_top_words(model, feature_names, n_top_words):
    norm = model.components_.sum(axis=1)[:, np.newaxis]
    for topic_idx, topic in enumerate(model.components_):
        print(80 * "-")
        print("Topic {}".format(topic_idx))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            print("{:.3f}".format(topic[i] / norm[topic_idx][0]) 
                  + '\t' + feature_names[i])

counts_feature_names = count_vect_model.get_feature_names()
n_top_words = 10
print_top_words(lda_model, counts_feature_names, n_top_words)