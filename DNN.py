# Utilities
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow.keras.metrics as metrics
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard

# Importing the dataset
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('assets/Train.csv',encoding=DATASET_ENCODING)

# Reading contractions.csv and storing it as a dict.
contractions = pd.read_csv('assets/contractions.csv', index_col='Contraction')
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
X_data, y_data = np.array(dataset['processed_text']), np.array(dataset['Sentiment'])

# Defining the model input length.
input_length = 60
tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
vocab_length = len(tokenizer.word_index) + 1
print("Tokenizer vocab length:", vocab_length)
cbow = KeyedVectors.load('assets/cbow')
X_data = pad_sequences(tokenizer.texts_to_sequences(X_data), maxlen=input_length)

print("X_train.shape:", X_data.shape)

Embedding_dimensions = 100
embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if cbow.wv.__contains__(word):
        embedding_matrix[token] = cbow.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

def getModel():
    embedding_layer = Embedding(input_dim = vocab_length, 
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix], 
                                input_length=input_length,
                                trainable=False)
    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    return model

training_model = getModel()
print(training_model.summary())

tensor_board = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=True, update_freq='epoch', profile_batch=2,
    embeddings_freq=0, embeddings_metadata=None
)

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_acc', min_delta=1e-6, patience=15),
             tensor_board
             ]

mymetrics=['acc',metrics.Precision(), metrics.Recall(), metrics.AUC(), metrics.RootMeanSquaredError()]

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=mymetrics)
history = training_model.fit(
    X_data, 
    y_data,
    batch_size=4096,
    epochs=50,
    validation_split=0.05,
    callbacks=callbacks,
    verbose=1
)

# Saving the tokenizer
with open('Tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)
save_model(training_model, 'ModelisSaved', )