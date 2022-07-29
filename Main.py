import tensorflow as tf
from SentimentAnalysis.preprocessing import DataPreprocessing
import NLTK, BERT, RNN
import pandas as pd
import os
import pickle
import sys
from app import app


if __name__ == '__main__':
    dataPrepro = DataPreprocessing.DataPreprocessing    
    cwd = os.getcwd()
    path=cwd + '\\data\\'
    fileName = 'AmazonReviews.csv'
    trainFile = 'train.csv'
    valFile = 'val.csv'
    modelPath = cwd + '\\SentimentAnalysis\\models\\'
    nltkClass = NLTK.NLTK
    bertClass = BERT.BERT
    rnnClass = RNN.RNN

    try:
        data = pd.read_csv(path+fileName)        
    except FileNotFoundError:
        print("File: {0} does not exist at".format(path+fileName))
        if dataPrepro._dsCount == 0:
            print("Downloading data...This would take time... Please wait...")
            data = dataPrepro.getData()
            dataPrepro.saveData(data,path+fileName)
        # Read data again from master file
        data = pd.read_csv(path+fileName)

    try:
        print("Reading Data from local drive... Hang in there... This might take some time...")
        trainData = pd.read_csv(path+trainFile)
        valData = pd.read_csv(path+valFile)  
    except FileNotFoundError:
        print("File: {0} does not exist at".format(path+trainFile))
        print("File: {0} does not exist at".format(path+valFile))
        print("Preprocessing and creating training data... This would take time... Please wait...")
        trainData, valData = dataPrepro.processData(data)
        dataPrepro.saveData(trainData,path+trainFile)
        dataPrepro.saveData(valData,path+valFile)
        # Read data again from train and val files
        trainData = pd.read_csv(path+trainFile)
        valData = pd.read_csv(path+valFile)

    #Call NLTK
    nltkClass.startProcessingNLTK(nltkClass, trainData, valData)
    #Call RNN
    rnnClass.startProcessingRNN(rnnClass, trainData, valData)
    #Call BERT
    bertClass.startProcessingBERT(bertClass, trainData, valData)

    app.run(port=12345, debug=True)

