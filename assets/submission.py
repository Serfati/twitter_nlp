# Utilities
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# Importing the dataset
DATASET_ENCODING = "ISO-8859-1"
test_df = pd.read_csv('Test.csv',encoding=DATASET_ENCODING)
dataset = pd.read_csv('Train.csv',encoding=DATASET_ENCODING)
dataset.head()

# Reading contractions.csv and storing it as a dict.
contractions = pd.read_csv('contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Defining regex patterns.
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_apply(tweet):

    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = tweet.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    tweet = re.sub(r'/', ' / ', tweet)
    return tweet

dataset['processed_text'] = dataset.SentimentText.apply(preprocess_apply)
test_df['processed_text'] = test_df.SentimentText.apply(preprocess_apply)

from sklearn.model_selection import train_test_split

X_data, y_data = np.array(dataset['processed_text']), np.array(dataset['Sentiment'])
X_test = np.array(test_df['processed_text'])

print('Data Split done.')

from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load('Word2Vec-twitter-100-dims')
print("Vocabulary Length:", len(word2vec_model.wv.vocab))

# Defining the model input length.
input_length = 60

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)

vocab_length = len(tokenizer.word_index) + 1
print("Tokenizer vocab length:", vocab_length)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_data), maxlen=input_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

model = tf.keras.models.load_model('LAST') 
print(model.summary())

y_pred=model.predict(X_test)
y_pred = np.where(y_pred>=0.5, 1, 0)
sub=pd.DataFrame(test_df['ID'])
sub['Sentiment']=y_pred
sub.to_csv('submission.csv',index=False)