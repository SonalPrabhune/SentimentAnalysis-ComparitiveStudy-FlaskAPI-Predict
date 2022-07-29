# Dependencies
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import NLTK, BERT, RNN

# Your API definition
app = Flask(__name__)
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    nltkClass = NLTK.NLTK
    rnnClass = RNN.RNN
    bertClass = BERT.BERT    
    try:
        #cwd = os.getcwd()
        #modelPath = cwd + '\\SentimentAnalysis\\models\\'
        
        jsonRequest = request.json
        print(jsonRequest)

        inputSentence = jsonRequest["body"]

        #testSentence = tokenizer.texts_to_sequences(inputSentence)
        #testSentence = pad_sequences(testSentence, maxlen=100, dtype='int32', value=0)
        #print(testSentence)
        predictionNLTK = nltkClass.testNLTK(nltkClass, inputSentence, *nltkClass.NLTKClassifiers)

        predictionRNN = rnnClass.testRNN(rnnClass, inputSentence)

        predictionBERT = bertClass.testBERT(bertClass, inputSentence)

        # loading
        #with open(modelPath+'tokenizer.pkl', 'rb') as handle:
        #    tokenizer = pickle.load(handle)
        #print("Tokenizer loaded")

        #model = load_model (modelPath+'RNNmodel.h5')
        #print ('Model loaded')  
        
        #sentiment = model.predict(testSentence,batch_size=1,verbose = 2)[0]
        #if(np.argmax(sentiment) == 0):
        #    prediction = "negative"
        #elif (np.argmax(sentiment) == 1):
        #    prediction = "positive"
        #print(prediction)
        return jsonify({'Prediction from the NLTK model': str(predictionNLTK),
                        'Prediction from the RNN model': str(predictionRNN),
                        'Prediction from the BERT model': str(predictionBERT)})

    except:
        return jsonify({'trace': traceback.format_exc()})
    