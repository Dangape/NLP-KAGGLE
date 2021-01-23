# Created by: Daniel Bemerguy 
# 22/01/2021 at 23:01
import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

#Import train and test dataset
train = pd.DataFrame(pd.read_csv('data/train.csv',sep = ','))
test = pd.DataFrame(pd.read_csv('data/test.csv',sep = ','))

stop_words = stopwords.words('english') #stop words lsit
tknzr = TweetTokenizer() #tokenizer object

#Remove stop words from tweets
def remove_stop_words(tweet):
    tokenized_tweet = tknzr.tokenize(tweet)
    processed_tweet = list(word for word in tokenized_tweet if word not in stop_words)
    return processed_tweet
test_tweet = train.loc[0,'text']
print(test_tweet)
print(remove_stop_words(test_tweet))

count_vectorizer = feature_extraction.text.CountVectorizer()