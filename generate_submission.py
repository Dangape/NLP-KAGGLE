# Created by: Daniel Bemerguy 
# 23/01/2021 at 13:10
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pandas as pd
from nltk.tokenize import word_tokenize

#Loading saved model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print('Loaded model from disk')

stop_words = stopwords.words('english') #stop words lsit
tknzr = TweetTokenizer() #tokenizer object


test_df = pd.DataFrame(pd.read_csv('data/test.csv',sep = ',')) #load test data

#test_df['processed_tweet'] = test_df['text'].apply(lambda x:remove_stop_words(x))

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words if word in useful_words])
    return my_dict

test_df["processed_tweet"] = test_df["text"].apply(lambda x:tknzr.tokenize(x))
test_df["processed_tweet"] = test_df["processed_tweet"].apply(lambda x:create_word_features(x))
test_df["target"] = test_df["processed_tweet"].apply(lambda x:loaded_model.classify(x))

print(test_df.head())

submission =test_df.loc[:,['id','target']]
print(len(submission))
print(submission)

submission.to_csv('submission.csv',index = False, header=True,encoding='utf-8',sep= ',')