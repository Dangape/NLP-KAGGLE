# Created by: Daniel Bemerguy 
# 26/01/2021 at 19:06
from gensim.models import Word2Vec,KeyedVectors
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Dropout
from keras.layers.embeddings import Embedding

train_df = pd.DataFrame(pd.read_csv('processed_train_data.txt', sep = ','))
X = train_df['processed_tweet']
Y = train_df['target']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size = 0.1)
total_tweets = X_train + X_train

#pad sequences so the vectors have the same lenght
max_lenght = max([len(s.split()) for s in total_tweets])

#define vocabulary size
vocab_size = len(tokenizer.word_index) + 1

X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens,maxlen=max_lenght,padding='post')
X_test_pad = pad_sequences(X_test_tokens,maxlen=max_lenght,padding='post')

embedding_dim = 100

print('Build model')
model = Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_lenght))
model.add(GRU(units = 32))
model.add(Dense(units = 32))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Training model')
model.fit(X_train_pad,y_train,batch_size=32,epochs=50,validation_data=(X_test_pad,y_test))