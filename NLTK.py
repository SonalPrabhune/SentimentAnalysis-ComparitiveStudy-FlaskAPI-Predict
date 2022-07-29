import pandas as pd
import numpy as np

import nltk
from nltk import classify
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.stem import *
from nltk.corpus import words

nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from statistics import mode
import random
import pickle
import os
import re

from SentimentAnalysis.preprocessing import DataPreprocessing

"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""

class NLTK(ClassifierI):   
    NLTKClassifiers = None
    vocabulary = list()
    cwd = os.getcwd()
    path=cwd + '\\data\\'
    modelPath = cwd + '\\SentimentAnalysis\\models\\'
    processedDataFile = cwd + '\\data\\ProcessedData.csv'
    dataPrepro = DataPreprocessing.DataPreprocessing 
    all_words = []
    documents = []
    word_features = []
    training_set = []
    testing_set = []

    test_sentence = "Wanted a simple and lightweight pc for my kids. This is perfect except the screen is just total garbage."
    #test_sentence = "The screen is just total garbage."
    #test_sentence = "This is a very good product. Great purchase"
        
    def startProcessingNLTK(self, inputTrainData, inputValData):
        trainedNBCClassifier = None
        trainedMNBClassifier = None
        trainedBernoulliNBClassifier = None
        trainedLogisticRegressionClassifier = None
        trainedLinearSVClassifier = None
        trainedSGDClassifier = None

        df = pd.concat([inputTrainData, inputValData], ignore_index=True)
        print(df)
        df['short_review'] = df['short_review'].apply(str)

        try:
            totalData = pd.read_csv(self.processedDataFile)        
        except FileNotFoundError:
            print("File: {0} does not exist at".format(self.processedDataFile))
            print("Preprocessing data...this might take some time")
            totalData = self.dataPrepro.dataPreprocessing(self.dataPrepro, df, self.processedDataFile)
            self.dataPrepro.saveData(totalData, self.processedDataFile)

        trainData, valData = np.split(totalData, [int(len(totalData)*0.8)])

        print(trainData.head())
        print(valData.head())

        positive_traindata = trainData[trainData["Sentiment"]==1]
        negative_traindata = trainData[trainData["Sentiment"]==0]  

        if (positive_traindata['short_review'].isnull().values.any()):
            positive_traindata = positive_traindata.dropna()

        if (negative_traindata['short_review'].isnull().values.any()):
            negative_traindata = negative_traindata.dropna()

        if (os.path.isfile(self.path+"NLTK_TaggedDocument.pkl")) and (os.path.isfile(self.path+"NLTK_AllWords.pkl")):
            with open(self.path+"NLTK_TaggedDocument.pkl",'rb') as handle:
                self.documents = pickle.load(handle)
            with open(self.path+"NLTK_AllWords.pkl","rb") as handle:
                self.all_words = pickle.load(handle)
        else:
            print("Creating Vocabulary...")
            self.getVocabulary(self,positive_traindata, negative_traindata)
            print("Vocabulary created")

        self.all_words = nltk.FreqDist(self.all_words)
                
        if (os.path.isfile(self.path+"NLTK_WordFeatures_5K.pkl")):
            with open(self.path+"NLTK_WordFeatures_5K.pkl",'rb') as handle:
                self.word_features = pickle.load(handle)
        else:
            self.word_features = list(self.all_words.keys())[:5000]
            with open(self.path+"NLTK_WordFeatures_5K.pkl","wb") as handle:
                pickle.dump(self.word_features, handle)

        #Loading models
        if (os.path.isfile(self.modelPath+"NLTK_OriginalNaiveBayes5k.pkl")):
            with open(self.modelPath+"NLTK_OriginalNaiveBayes5k.pkl",'rb') as handle:
                trainedNBCClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training NLTK Naive Bayes Classifier...")
            trainedNBCClassifier = nltk.NaiveBayesClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_OriginalNaiveBayes5k.pkl","wb") as handle:
                pickle.dump(trainedNBCClassifier, handle)

            print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(trainedNBCClassifier, self.testing_set))*100)
            trainedNBCClassifier.show_most_informative_features(15)

            print("Running diagnostics using NLTK Naive Bayes Classifier...")
            self.runDiagnostics(self, valData, trainedNBCClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using NLTK Naive Bayes Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedNBCClassifier)
            print(testResults)

    
        if (os.path.isfile(self.modelPath+"NLTK_MNB_classifier5k.pkl")):
            with open(self.modelPath+"NLTK_MNB_classifier5k.pkl",'rb') as handle:
                trainedMNBClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training NLTK Multinomial Naive Bayes Classifier...")
            trainedMNBClassifier = SklearnClassifier(MultinomialNB())
            trainedMNBClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_MNB_classifier5k.pkl","wb") as handle:
                pickle.dump(trainedMNBClassifier, handle)

            print("Multinomial Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(trainedMNBClassifier, self.testing_set))*100)
            
            print("Running diagnostics using Multinomial Naive Bayes Classifier...")
            self.runDiagnostics(self, valData, trainedMNBClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using Multinomial Naive Bayes Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedMNBClassifier)
            print(testResults)
        


        if (os.path.isfile(self.modelPath+"NLTK_BernoulliNB_classifier5k.pkl")):
            with open(self.modelPath+"NLTK_BernoulliNB_classifier5k.pkl",'rb') as handle:
                trainedBernoulliNBClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training NLTK Bernoulli Naive Bayes Classifier...")
            trainedBernoulliNBClassifier = SklearnClassifier(BernoulliNB())
            trainedBernoulliNBClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_BernoulliNB_classifier5k.pkl","wb") as handle:
                pickle.dump(trainedBernoulliNBClassifier, handle)

            print("Bernoulli Naive Bayes Classifier accuracy percent:", (nltk.classify.accuracy(trainedBernoulliNBClassifier, self.testing_set))*100)
            
            print("Running diagnostics using Bernoulli Naive Bayes Classifier...")
            self.runDiagnostics(self, valData, trainedBernoulliNBClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using Bernoulli Naive Bayes Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedBernoulliNBClassifier)
            print(testResults)
        
   

        if (os.path.isfile(self.modelPath+"NLTK_LogisticRegression_classifier5k.pkl")):
            with open(self.modelPath+"NLTK_LogisticRegression_classifier5k.pkl",'rb') as handle:
                trainedLogisticRegressionClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training NLTK Logistic Regression Classifier...")
            trainedLogisticRegressionClassifier = SklearnClassifier(LogisticRegression())
            trainedLogisticRegressionClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_LogisticRegression_classifier5k.pkl","wb") as handle:
                pickle.dump(trainedLogisticRegressionClassifier, handle)

            print("Logistic Regression Classifier accuracy percent:", (nltk.classify.accuracy(trainedLogisticRegressionClassifier, self.testing_set))*100)
            
            print("Running diagnostics using Logistic Regression Classifier...")
            self.runDiagnostics(self, valData, trainedLogisticRegressionClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using Logistic Regression Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedLogisticRegressionClassifier)
            print(testResults)

  
         
        if (os.path.isfile(self.modelPath+"NLTK_LinearSVC_classifier5k.pkl")):
            with open(self.modelPath+"NLTK_LinearSVC_classifier5k.pkl",'rb') as handle:
                trainedLinearSVClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training NLTK Linear Support Vector Classifier...")
            trainedLinearSVClassifier = SklearnClassifier(LinearSVC())
            trainedLinearSVClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_LinearSVC_classifier5k.pkl","wb") as handle:
                pickle.dump(trainedLinearSVClassifier, handle)

            print("Linear Support Vector Classifier accuracy percent:", (nltk.classify.accuracy(trainedLinearSVClassifier, self.testing_set))*100)
            
            print("Running diagnostics using Linear Support Vector Classifier...")
            self.runDiagnostics(self, valData, trainedLinearSVClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using Linear Support Vector Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedLinearSVClassifier)
            print(testResults)


        if (os.path.isfile(self.modelPath+"NLTK_SGDC_classifier5k.pkl")):
            with open(self.modelPath+"NLTK_SGDC_classifier5k.pkl",'rb') as handle:
                trainedSGDClassifier = pickle.load(handle)
        else:
            if (self.training_set==[] or self.testing_set==[]):
                self.training_set, self.testing_set = self.get_train_test_set(self)

            print("Training Stochastic Gradient Descent Classifier...")
            trainedSGDClassifier = SklearnClassifier(SGDClassifier())
            trainedSGDClassifier.train(self.training_set)
            with open(self.modelPath+"NLTK_SGDC_classifier5k.pkl","wb") as handle:
                pickle.dump(trainedSGDClassifier, handle)

            print("Stochastic Gradient Descent Classifier accuracy percent:", (nltk.classify.accuracy(trainedSGDClassifier, self.testing_set))*100)
            
            print("Running diagnostics using Stochastic Gradient Descent Classifier...")
            self.runDiagnostics(self, valData, trainedSGDClassifier)
            print("Running NLTK Sentiment Analysis on test sentence using Stochastic Gradient Descent Classifier")        
            print(self.test_sentence)
            testResults = self.testNLTK(self, self.test_sentence, trainedSGDClassifier)
            print(testResults)


        self.getClassifiers(self,trainedNBCClassifier,
                            trainedMNBClassifier,
                            trainedBernoulliNBClassifier,
                            trainedLogisticRegressionClassifier,
                            trainedLinearSVClassifier,
                            trainedSGDClassifier)

        #print("Running diagnostics using all NLTK and SkLearn Classifiers...")
        #self.runDiagnostics(self, valData, *self.NLTKClassifiers)

        print("Running NLTK Sentiment Analysis on test sentence using all NLTK and SkLearn Classifiers")
        testResults, confidence = NLTK.currentTrainedClassifier = self.testNLTK(self, self.test_sentence, *self.NLTKClassifiers)
        print("Sentiment Analysis Result = " + testResults + ", with confidence from all classifiers of " + str(confidence))

        

    def getClassifiers(self, *classifiers):
        self.NLTKClassifiers = classifiers


    def get_train_test_set(self):
        print("Creating Features...")
        featuresets = [(self.extract_features(self, rev), category) for (rev, category) in self.documents]

        random.shuffle(featuresets)
        print("Features created. Total size of features is...")
        print(len(featuresets))

        self.testing_set = featuresets[10000:]
        self.training_set = featuresets[:10000]

        return self.training_set, self.testing_set


    def getVocabulary(self, positive_traindata, negative_traindata):
        #  j is adject, r is adverb, and v is verb
        #allowed_word_types = ["J","R","V"]
        #allowed_word_types = ["J", "JR", "JJS","R","V", "RB", "RBR", "RBS", "RP"]
        allowed_word_types = ["J"]

        for p in positive_traindata['short_review']:
            self.documents.append( (p, "Positive") )
            words = word_tokenize(p)
            pos = nltk.pos_tag(words)
            for w in pos:
                if (w[1][0] in allowed_word_types) and (w[1][0] not in self.all_words):
                    self.all_words.append(w[0].lower())

    
        for p in negative_traindata['short_review']:
            self.documents.append( (p, "Negative") )
            words = word_tokenize(p)
            pos = nltk.pos_tag(words)
            for w in pos:
                if (w[1][0] in allowed_word_types) and (w[1][0] not in self.all_words):
                    self.all_words.append(w[0].lower())


        with open(self.path+"NLTK_TaggedDocument.pkl","wb") as handle:
            pickle.dump(self.documents, handle)

        with open(self.path+"NLTK_AllWords.pkl","wb") as handle:
            pickle.dump(self.all_words, handle)
  
    

    def extract_features(self, review):
        words = word_tokenize(review)
        features={}              

        for word in self.word_features:
            features[word] = (word in words)
        return features


    def getNBCSentimentCalculator(self, review, *classifiers):
        votes = []
        conf = 0.0
        result = ""
        for c in classifiers:
            v = c.classify(self.extract_features(self, review))
            votes.append(v)
        if not votes == []:
            result = mode(votes)
            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
        return result, conf

    
    def testNLTK(self, review, *classifiers):
        d = {'short_review': [review], 'Sentiment': [0]}
        df = pd.DataFrame(data=d)
        totalData = self.dataPrepro.dataPreprocessing(self.dataPrepro, df, self.processedDataFile)
        test_sentence =  totalData['short_review'][0]
        print(test_sentence)

        testResults, conf = self.getNBCSentimentCalculator(self, test_sentence, *classifiers)
        return testResults, conf      

    def runDiagnostics(self, valdata, *classifiers):
        if (valdata['short_review'].isnull().values.any()):
            valdata = valdata.dropna()

        testResults = [self.getNBCSentimentCalculator(self, review, *classifiers) for review in valdata["short_review"]]
                
        positive_testdata = valdata[valdata["Sentiment"]==1]
        negative_testdata = valdata[valdata["Sentiment"]==0]

        positiveReviewsResult = [self.getNBCSentimentCalculator(self, review, *classifiers) for review in positive_testdata["short_review"]]
        negativeReviewsResult = [self.getNBCSentimentCalculator(self, review, *classifiers) for review in negative_testdata["short_review"]]

        numPosResults = [x for x in positiveReviewsResult]
        numNegResults = [x for x in negativeReviewsResult]
        true_positive = sum(x[0] == 'Positive' for x in numPosResults)
        true_negative = sum(x[0] == 'Negative' for x in numNegResults)
        pctTruePositive = float(true_positive) / len(positiveReviewsResult)
        pctTrueNegative = float(true_negative) / len(negativeReviewsResult)
        neg_cnt = 0
        pos_cnt = 0
        for res in testResults:
            if(res[0] == 'Negative'): 
                neg_cnt = neg_cnt + 1
            if(res[0] == 'Positive'): 
                pos_cnt = pos_cnt + 1

        print('[Negative]: %s '  % (neg_cnt))        
        print('[Positive]: %s '  % (pos_cnt))   

        totalAccurate = float(true_positive) + float(true_negative)
        total = len(positiveReviewsResult) + len(negativeReviewsResult)
    
        print ("Accuracy on positive reviews = " + "%.2f"% (pctTruePositive*100))
        print ("Accuracy on negative reviews = " + "%.2f"% (pctTrueNegative*100))
        print ("Overall Accuracy = " + "%.2f"% (totalAccurate*100/total))
