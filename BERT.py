import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import pandas as pd
import numpy as np
import os
import pickle

class BERT:
    modelPath = os.getcwd() + '\\SentimentAnalysis\\models\\'

    def __init__(self, data):        
        #num_gpus_available = len(tf.config.experimental.list_physical_devices('GPU'))
        #print("Num GPUs Available: ", num_gpus_available)
        #print(assert num_gpus_available > 0)
        self.startProcessingBERT(data)
        
    def startProcessingBERT(self, trainData, valData):        
        training_sentences = trainData['short_review'].astype('str').values.tolist()
        print(training_sentences[0])
        training_labels = trainData.Sentiment.astype('int64').tolist()
        print(training_labels[0])

        #validation_sentences = valData.short_review.astype('string').tolist()
        validation_sentences = valData['short_review'].astype('str').values.tolist()
        print(validation_sentences[0])
        validation_labels = valData.Sentiment.astype('int64').tolist()
        print(validation_labels[0])

        # Tokenization of text and conversion into tokens
        # Tokenizer class from pre-trained DistilBert
        # Other options (not used here) - Bag of words, TFIDF, Tokenizer from Keras, Word Embedding
        # Assign tokenizer object to the tokenizer class  
        try:
            #with open(self.modelPath+'tokenizerBERT.pkl', 'rb') as handle:
            #    tokenizer = pickle.load(handle)
            tokenizer = DistilBertTokenizerFast.from_pretrained(self.modelPath+"bertTokenizer", local_files_only=True)
            print("BERT Tokenizer loaded")
        except:
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            print(tokenizer([training_sentences[0]], truncation=True,
                                        padding=True, max_length=128))

            # saving
            tokenizer.save_pretrained(self.modelPath+"bertTokenizer")
            #with open(self.modelPath+'tokenizerBERT.pkl', 'wb') as handle:
            #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        

        # Use TFDistilBertForSequenceClassification for the sentiment analysis 
        # and put the "num-labels" parameter equal to 2 as we are doing a binary classification
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)

        # We don't need to put any additional layers and using a Hugging face transformer, 
        # we can now train our model with the following configuration:
        # epochs: 2
        # Batch size: 16 
        # Training set batch size: 8 
        # Learning rate (Adam): 5e-5 (0.00005)
        # The number of epochs can be increased, however, it will give rise to overfitting problems as well as take more time 
        # for the model to train. The complete model gets trained in around 2hrs, that's why it is important to keep 
        # the number of epochs and batch size low.
        try:
            #with open(self.modelPath+'tokenizerBERT.pkl', 'rb') as handle:
            #    tokenizer = pickle.load(handle)
            #tokenizer = DistilBertTokenizerFast.from_pretrained(self.modelPath+"bertTokenizer", local_files_only=True)
            #print("BERT Tokenizer loaded")
            loaded_model = TFDistilBertForSequenceClassification.from_pretrained(self.modelPath+"BERTsentiment")
            print ('BERT Model loaded')
        except:
            # Using "from-Tensor-Slices", we can easily combine our features tokens and labels into a dataset.
            train_encodings = tokenizer(training_sentences,
                                        truncation=True,
                                        padding=True)
            val_encodings = tokenizer(validation_sentences,
                                        truncation=True,
                                        padding=True)
            train_dataset = tf.data.Dataset.from_tensor_slices((
                                        dict(train_encodings),
                                        training_labels
                                        ))
            val_dataset = tf.data.Dataset.from_tensor_slices((
                                        dict(val_encodings),
                                        validation_labels
                                        ))
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
            model.compile(optimizer=optimizer, 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
            model.fit(train_dataset.shuffle(100).batch(8),
                        epochs=2,
                        batch_size=16,
                        validation_data=val_dataset.shuffle(100).batch(16))

            model.save_pretrained(self.modelPath+"BERTsentiment")    
            
            # Evaluation
            # To evaluate our model accuracy on unseen data, we can load the saved data and test it on new sentences 
            # and see if the sentiment is predicted correctly or not
            test_sentence = "Wanted a simple and lightweight pc for my kids. This is perfect except the screen is just total garbage."
            print(test_sentence)
            testResults = self.testBERT(self, test_sentence)
            print(testResults)

    def testBERT(self, test_sentence):
        try:
            with open(self.modelPath+'tokenizerBERT.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
                print("BERT Tokenizer loaded")
            loaded_model = TFDistilBertForSequenceClassification.from_pretrained(self.modelPath+"BERTsentiment")
            print ('BERT Model loaded')
            print("Evaluating Model...")
            
            predict_input = tokenizer.encode(test_sentence,
                                                truncation=True,
                                                padding=True,
                                                return_tensors="tf")

            tf_output = loaded_model.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1)
            print(tf_prediction)
            labels = ['Negative','Positive']
            label = tf.argmax(tf_prediction, axis=1)
            print(label)
            label = label.numpy()
            return(labels[label[0]])
        except:
            return("Failed to load BERT model and / or tokenizer")

