# Created by: Daniel Bemerguy 
# 26/01/2021 at 19:06
from gensim.models import Word2Vec,KeyedVectors
import pandas as pd
import numpy as np
import gensim
from itertools import islice
from sklearn import model_selection
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
train_df = pd.DataFrame(pd.read_csv('processed_train_data.txt', sep = ','))
lines = train_df['processed_tweet'].values.tolist()

X = train_df['processed_tweet']
Y = train_df['target']

#using a word2vec model pretrained by google
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
model.init_sims(replace=True)

#Check model
print(list(islice(model.vocab, 13030, 13050)))
print(model.wv.most_similar(('horrible')))

#Saving model
filename = 'word2vec_model.txt'
model.wv.save_word2vec_format(filename,binary=False)
print('Model saved to disk')
