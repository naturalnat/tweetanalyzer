import pandas as pd   
import numpy as np  
import re
from functools import reduce
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams 

import matplotlib.pyplot as plt
import seaborn as sns


tweets = pd.read_csv('tweets.csv')  

#descending sort
tweets = tweets.sort_values(by=['engagements'], ascending=False)

#lowercase text
tweets['Tweet text'] = tweets['Tweet text'].str.lower()


#remove stop words
stop_words = set(stopwords.words('english')) 
    
tweets['Tweet text'] = [' '.join([w for w in x.lower().split() if w not in stop_words]) 
    for x in tweets['Tweet text'].tolist()]

#remove username, links, hashtags
def rm_user_links(tweets):
    tweets = re.sub('@[^\s]+','',tweets)
    tweets = re.sub('http[^\s]+','',tweets)
    tweets = re.sub('#[^\s]+','',tweets)
    return tweets
tweets['Tweet text'] = tweets['Tweet text'].apply(rm_user_links)

#remove punctuations
tweets['Tweet text'] = tweets['Tweet text'].str.replace("[^a-zA-Z#]", " ")

#remove words with len < 2
tweets['Tweet text'] = tweets['Tweet text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



#tokenization
tokenized_tweets = tweets['Tweet text'].apply(lambda x: list(ngrams(x.split(), 2)))

display(tokenized_tweets)

#count words
l = reduce(lambda x, y: list(x)+list(y), zip(tokenized_tweets))
flatten = [item for sublist in l for item in sublist]
counts = Counter(flatten).most_common()
df = pd.DataFrame.from_records(counts, columns=['Phrase', 'Count'])
df['Phrase'] = df['Phrase'].apply(lambda x: ' '.join([w for w in x]))



#show graph

df = df.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(15,4))
ax = sns.barplot(data=df, x= "Phrase", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

