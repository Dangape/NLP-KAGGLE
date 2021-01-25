# Created by: Daniel Bemerguy 
# 22/01/2021 at 23:01
import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pickle
from nltk.classify import NaiveBayesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

#Import train and test dataset
train_df = pd.DataFrame(pd.read_csv('data/train.csv',sep = ','))


stop_words = stopwords.words('english') #stop words lsit
tknzr = TweetTokenizer() #tokenizer object

#Remove stop words from tweets
def remove_stop_words(tweet):
    tokenized_tweet = tknzr.tokenize(tweet)
    processed_tweet = list(word for word in tokenized_tweet if word not in stop_words)
    processed_tweet = ' '.join(processed_tweet)
    return processed_tweet

train_df['processed_tweet'] = train_df['text'].apply(lambda x:remove_stop_words(x))


#Define feature vector containing words excluding stop words and punctuation
feature_vector = []
punctuation = ['.',':',"'",'-',';','...',',','/','..','(',')']

for i in range(0,len(train_df)):
    feature_vector.extend(list(word for word in tknzr.tokenize(train_df.loc[i,'processed_tweet']) if word not in punctuation))

print(len(train_df))
document = train_df.loc[:,['processed_tweet','target']]

#Function to search for the words of the review in the feature_vector
def find_feature(word_list):
    feature = {}
    for x in feature_vector:
        feature[x] = x in word_list

    return(feature)

document = document.values.tolist()
#########################
#Naive Bayes classifier
# feature_sets = [(find_feature(word_list),target) for (word_list,target) in document]
#
# #train_set,test_set = model_selection.train_test_split(feature_sets,test_size = 0.0)
#
# model = NaiveBayesClassifier.train(feature_sets)
#
# # accuracy = nltk.classify.accuracy(model, test_set)
# # print('SVC Accuracy : {}%'.format(accuracy*100))
##################################
#Pipeline
X = train_df.loc[:,'processed_tweet']
Y = train_df.loc[:,'target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size = 0.25)
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, tol=None)),
               ])
sgd.fit(X, Y)

y_pred = sgd.predict(X_test)

print(y_pred[0])
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred,target_names=['1','0']))
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(sgd, open(filename, 'wb'))
pickle.dump(feature_vector, open("feature_vector.pickle","wb"))
print('Saved model to disk')

