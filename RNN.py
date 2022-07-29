import tensorflow as tf
import numpy as np
import pandas as pd 
import io
import json
import math

from SentimentAnalysis.preprocessing import DataPreprocessing

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.preprocessing.text import tokenizer_from_json

import matplotlib.pyplot as plt

from numpy import array
from numpy import asarray
from numpy import zeros

import pickle
import os

class RNN:
    cwd = os.getcwd()
    path=cwd + '\\data\\'
    modelPath = cwd + '\\SentimentAnalysis\\models\\'
    processedDataFile = cwd + '\\data\\ProcessedData.csv'
    dataPrepro = DataPreprocessing.DataPreprocessing   

    def plot_graphs(history, metric):
      plt.plot(history.history[metric])
      plt.plot(history.history['val_'+metric], '')
      plt.xlabel("Epochs")
      plt.ylabel(metric)
      plt.legend([metric, 'val_'+metric])

    def startProcessingRNN(self, trainData, valData):
        df = pd.concat([trainData, valData], ignore_index=True)
        print(df)
        df['short_review'] = df['short_review'].apply(str)
        
        #Text Preprocessing
        try:
            totalData = pd.read_csv(self.processedDataFile)        
        except FileNotFoundError:
            print("File: {0} does not exist at".format(self.processedDataFile))
            print("Preprocessing data...this might take some time")
            totalData = self.dataPrepro.dataPreprocessing(self.dataPrepro, df, self.processedDataFile)
            self.dataPrepro.saveData(totalData, self.processedDataFile)
        #training_sentences, val_sentences, testing_sentences = np.split(corpus, [int(len(corpus)*0.8), int(len(corpus)*0.9)])

        if (totalData['short_review'].isnull().values.any()):
            totalData = totalData.dropna()

        training_sentences, val_sentences, testing_sentences = np.split(totalData['short_review'], 
                                                                        [int(len(totalData['short_review'])*0.8), 
                                                                         int(len(totalData['short_review'])*0.9)])

        print(len(training_sentences))
        print(len(val_sentences))
        print(len(testing_sentences))

        #Define the hyperparameters for tokenization
        vocab_size = 20000
        embedding_dim = 100
        max_length = 100
        trunc_type='post'
        padding_type='post'
        oov_tok = "<OOV>"
        training_size = 70000
        test_size = 95000


        if os.path.isfile(self.modelPath+'RNNtokenizer.json'):
            tokenizer = self.loadTokenizer(self.modelPath)
        else:
            #Convert text into tokens
            tokenizer = Tokenizer(num_words=vocab_size, lower = True, oov_token=oov_tok)
            tokenizer.fit_on_texts(training_sentences)

            # Adding 1 because of reserved 0 index
            vocab_size = len(tokenizer.word_index) + 1

            training_sequences = tokenizer.texts_to_sequences(training_sentences)
            training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            val_sequences = tokenizer.texts_to_sequences(val_sentences)
            val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
            testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            print(len(df['Sentiment']))
            training_label, val_label, testing_label = np.split(df['Sentiment'], 
                                                                [int(len(df['Sentiment'])*0.8), int(len(df['Sentiment'])*0.9)])

            training_label = np.reshape(training_label.array, (len(training_label),1)).astype('int32')
            val_label = np.reshape(val_label.array, (len(val_label),1)).astype('int32')
            testing_label = np.reshape(testing_label.array, (len(testing_label),1)).astype('int32')

            print(len(training_label))
            print(len(val_label))
            print(len(testing_label))


            embeddings_dictionary = dict()
            glove_file = open(self.path+'\glove.6B\glove.6B.100d.txt', encoding="utf8")

            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = asarray(records[1:], dtype='float32')
                embeddings_dictionary [word] = vector_dimensions
            glove_file.close()

            embedding_matrix = zeros((vocab_size, 100))
            for word, index in tokenizer.word_index.items():
                embedding_vector = embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
       
            batch_size = 32
        
            #if os.path.isfile(self.modelPath+'RNNmodel.h5'):
            #    model = load_model(self.modelPath+'RNNmodel.h5', compile=True)
            if os.path.isfile(self.modelPath+'RNNmodel.h5'):
                model = self.loadModel(self.modelPath)
            else:
                tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                            write_graph=True, write_images=False)

                #RNNmodel_withEmbeddings_noFlatten.h5
                model = tf.keras.Sequential([                             
                    #tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                    tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], 
                                                input_length=max_length , trainable=False),
                    tf.keras.layers.Dropout(0.4),
                    #tf.keras.layers.LSTM(embedding_dim,dropout=0.2, recurrent_dropout=0.2,return_sequences=True),
                    #tf.keras.layers.Flatten(),                         
                    #tf.keras.layers.GlobalAveragePooling1D(),
                    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                    #tf.keras.layers.Dense(64, activation='relu'),
                    #tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    #tf.keras.layers.Dense(8, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
            

                #RNNmodel_smaller_without_gloveEmbedings.h5
                #vocabulary_size = len(tokenizer.word_counts.keys())+1
                #max_words = 100
                #embedding_size = 32
                #model = Sequential()
                #model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
                #model.add(LSTM(200))
                #model.add(Dense(1, activation='sigmoid'))
                #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
                model.summary()
                model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
                history = model.fit(training_padded, training_label, epochs = 10, 
                                    batch_size=batch_size, verbose = 1, 
                                    validation_data=(val_padded, val_label),callbacks=[tensorboard])

                self.saveModels(model, tokenizer, self.modelPath)

                print("Evaluating Model...")        
                test_loss, test_acc = model.evaluate(testing_padded, testing_label, verbose=True)
                print('Test Loss:', test_loss)
                print('Test Accuracy:', test_acc)

                self.plot_graphs (history,'accuracy')
                
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])

                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train','test'], loc='upper left')
                plt.show()

                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])

                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train','test'], loc='upper left')
                plt.show()

            test_sentence = ["Wanted a simple and lightweight pc for my kids. This is perfect except the screen is just total garbage."] 
            #test_sentence = ["Wanted a simple and lightweight pc for my kids. The screen is just total garbage."] 
            #test_sentence = ["The product was really good. Everything worked as expected"]
            #test_sentence = ["Bad product. This has been the worst experience"]
            #test_sentence = ["Perfect"]
            print(self.testRNN(self, test_sentence))

    

    def testRNN(self, input_sentence):
        try:
            model = self.loadModel(self.modelPath)
            print ('RNN Model loaded')  
            tokenizer = self.loadTokenizer(self.modelPath)
            print("RNN Tokenizer loaded")

            d = {'short_review': [input_sentence], 'Sentiment': [0]}
            df = pd.DataFrame(data=d)
            totalData = self.dataPrepro.dataPreprocessing(self.dataPrepro, df, self.path+'ProcessedData.csv')
            test_sentence =  totalData['short_review'][0]
            print(test_sentence)
            
            instance = tokenizer.texts_to_sequences(test_sentence)

            flat_list = []
            for sublist in instance:
                for item in sublist:
                    flat_list.append(item)

            flat_list = [flat_list]

            instance = pad_sequences(flat_list, padding='post', maxlen=100)

            pred_prob =  model.predict(instance)
            print(pred_prob)

            if((pred_prob > 0.6).astype('int32') == 0):
                return("Negative. Probability = {0}".format(pred_prob))
            elif ((pred_prob > 0.6).astype('int32') == 1):
                return("Positive. Probability = {0}".format(pred_prob))
            
        except:
            return("Failed to load RNN model and / or tokenizer")
        

    def saveModels(model, tokenizer, modelPath):
        # Save the trained weights
        model.save_weights(modelPath+'RNNmodel_weights.h5')

        # Save the model architecture
        with open(modelPath+'RNNmodel_architecture.json', 'w') as f:
            f.write(model.to_json())

        # Save the tokenizer
        with open(modelPath+'RNNtokenizer.json', 'w') as f:
            f.write(tokenizer.to_json())

    def loadModel(modelPath):        
        # Model reconstruction from JSON file
        with open(modelPath+'RNNmodel_architecture.json', 'r') as f:
            model = model_from_json(f.read())

        # Load weights into the new model
        model.load_weights(modelPath+'RNNmodel_weights.h5')

        return model

    def loadTokenizer(modelPath):
        with open(modelPath+'RNNtokenizer.json') as f:
            tokenizer = tokenizer_from_json(f.read())

        return tokenizer