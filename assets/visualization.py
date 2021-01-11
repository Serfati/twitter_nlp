# Utilities
import pandas as pd

# Plot libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Importing the dataset
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('Train.csv',encoding=DATASET_ENCODING)
dataset.head()

plt.figure()
plt.hist(dataset['SentimentText'].str.split().apply(len).value_counts())
plt.xlabel('number of words in sentence')
plt.ylabel('frequency')
plt.title('Words occurrence frequency')
plt.show()

s = dataset['Sentiment'].value_counts()
s = (s/s.sum())*100

plt.figure()
bars = plt.bar(s.index, s.values, color = ['green', 'red'], alpha = .6)
plt.xticks(s.index, ['Positive', 'Negative'], fontsize = 15)
plt.tick_params(bottom = False, top = False, left = False, right = False, labelleft = False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, s = str(bar.get_height())[:2] + '%', ha = 'center', fontsize = 15)
plt.title('Tweet polarity', fontsize = 17)
plt.show()

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def remove_sw(review):
    tokens = word_tokenize(review)
    tokens = [w for w in tokens if not w in stop_words]
    return " ".join(tokens)

dataset["SentimentText"] = dataset["SentimentText"].apply(remove_sw)


import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

neg = dataset[dataset['Sentiment'] == 0]
pos = dataset[dataset['Sentiment'] == 1]


wc = WordCloud(max_words = 1000 , width = 1600 ,background_color="white", height = 800,
              collocations=False).generate(" ".join(pos.SentimentText))
plt.figure(figsize = (20,20))
plt.imshow(wc)
plt.title('WordCloud for Positive Tweets')
plt.axis("off")
plt.show()

wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(neg.SentimentText))
plt.figure(figsize = (20,20))
plt.imshow(wc)
plt.title('WordCloud for Negative Tweets')
plt.axis("off")
plt.show()

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

most_common_uni = get_top_text_ngrams(neg.SentimentText,10,1)
most_common_uni = dict(most_common_uni)
temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
temp["Common_words"] = list(most_common_uni.keys())
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp, x="Count", y="Common_words", title='Unigram - Commmon Words in Negative Tweet', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()

most_common_uni = get_top_text_ngrams(pos.SentimentText,10,1)
most_common_uni = dict(most_common_uni)
temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
temp["Common_words"] = list(most_common_uni.keys())
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp, x="Count", y="Common_words", title='Unigram - Commmon Words in Positive Tweet', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()

most_common_uni = get_top_text_ngrams(neg.SentimentText,10,2)
most_common_uni = dict(most_common_uni)
temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
temp["Common_words"] = list(most_common_uni.keys())
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp, x="Count", y="Common_words", title='Bigram - Commmon Words in Negative Tweet', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()

most_common_uni = get_top_text_ngrams(pos.SentimentText,10,2)
most_common_uni = dict(most_common_uni)
temp = pd.DataFrame(columns = ["Common_words" , 'Count'])
temp["Common_words"] = list(most_common_uni.keys())
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp, x="Count", y="Common_words", title='Bigram - Commmon Words in Positive Tweet', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()

