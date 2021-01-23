# Created by: Daniel Bemerguy 
# 22/01/2021 at 23:01
import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer


#Import train and test dataset
train = pd.DataFrame(pd.read_csv('data/train.csv',sep = ','))
test = pd.DataFrame(pd.read_csv('data/test.csv',sep = ','))

stop_words = stopwords.words('english') #stop words lsit
tknzr = TweetTokenizer() #tokenizer object

#Remove stop words from tweets
def remove_stop_words(tweet):
    tokenized_tweet = tknzr.tokenize(tweet)
    processed_tweet = list(word for word in tokenized_tweet if word not in stop_words)
    processed_tweet = ' '.join(processed_tweet)
    return processed_tweet

train['processed_tweet'] = train['text'].apply(lambda x:remove_stop_words(x))

test = train.loc[0,'processed_tweet']
#Define feature vector containing words excluding stop words and punctuation
feature_vector = []
punctuation = ['.',':',"'",'-',';','...',',','/','..','(',')']

for i in range(0,len(train)):
    feature_vector.extend(list(word for word in tknzr.tokenize(train.loc[i,'processed_tweet']) if word not in punctuation))
print(feature_vector)

