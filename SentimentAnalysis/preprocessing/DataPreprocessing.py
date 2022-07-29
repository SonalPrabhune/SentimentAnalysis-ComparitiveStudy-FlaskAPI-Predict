import tensorflow as tf
import pandas as pd
import numpy as np

import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.stem import *
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import ast
        
class DataPreprocessing: 
    _dsCount = 0
    df = pd.DataFrame()
    
    def __init__(self):
      if self._dsCount==0:
          #To make sure that the clall to getData is made only once
          self._dsCount += 1
      
    def getData():
         print("Loading Amazon Reviews data... Please wait...")
         ds = tfds.load('amazon_us_reviews/Mobile_Electronics_v1_00', split='train', shuffle_files=True)
         assert isinstance(ds, tf.data.Dataset)
         #convert the dataset into a pandas dataframe
         df = tfds.as_dataframe(ds)
         return df

    def saveData(df, file):        
        df.to_csv(file)
        print('File saved')

    def processData(df):
        #The rating provided by the customer is on a scale of 1-5( 5 being the highest). 
        #As we are going to implement a binary classification model, we will need to convert these ratings into 2 categories,
        #i.e 1 and 0. Ratings above 3 will be labeled as Positive(1) and below or equal to 3 will be negative(0). 
        #The following code will help us implement these steps.
        df["Sentiment"] = df["data/star_rating"].apply(lambda score: "positive" if score >= 3 else "negative")
        df['Sentiment'] = df['Sentiment'].map({'positive':1, 'negative':0})
        df['short_review'] = df['data/review_body'].map(lambda v: ast.literal_eval(v).decode())
        df = df[["short_review", "Sentiment"]]

        reviews = df['short_review'].values.tolist()
        labels = df['Sentiment'].tolist()

        training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(reviews, labels, test_size=.2)
        
        trainData = pd.DataFrame(data = np.stack((training_sentences, training_labels), axis=1),columns=['short_review', 'Sentiment'])
        trainData['Sentiment'] = pd.to_numeric(trainData['Sentiment'])

        valData = pd.DataFrame(data = np.stack((validation_sentences, validation_labels), axis=1),columns=['short_review', 'Sentiment'])
        valData['Sentiment'] = pd.to_numeric(valData['Sentiment'])

        return trainData, valData

    def dataPreprocessing(self, df, filePath):        
        corpus = []
        for i in range(0, len(df)):            
            review = re.sub('[^a-zA-Z]', ' ', str(df['short_review'][i]))
            review = review.lower()
            review = review.split()

            #stop_words = set(stopwords.words('english'))
 
            #word_tokens = word_tokenize(review)
            #for w in word_tokens:
            #    if w not in stop_words:
            #        w = ''.join(w)
            #        corpus.append(w)
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)
        df_corpus = pd.DataFrame(data=corpus, columns = ['short_review'])        
        totalData = pd.concat([df_corpus['short_review'],df['Sentiment']], axis=1, ignore_index=True)
        totalData.columns = ['short_review', 'Sentiment']    
        if (totalData['short_review'].isnull().values.any()):
            totalData = totalData.dropna()
        return totalData