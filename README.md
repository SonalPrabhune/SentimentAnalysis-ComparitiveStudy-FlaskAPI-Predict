Code Implementation
The code is structured such that the Main.py is the starting point. This is where the application starts. The Main.py will do the following…
•	Check if there exists a processed .csv data file. If not, it will call the DataProcessing.py code’s respective functions to preprocess the data. 
•	Then it will split the data into Train and Test Data as 60% train data and 40% test data and save it to corresponding .csv files on disk
•	Then it will execute each of the 3 comparative models NLTK (Python’s Sci-kit Learn Natural Language Tool Kit) from which I have used 6 different models, the RNN Recurrent Neural Network model built from scratch using TensorFlow 2.0 and finally BERT (Bidirectional Encoder Representations from Transformers) which is an open-source pre-trained model provided by Hugging Face. 
•	Finally, the app.py is called which invokes a Flask api Endpoint for POST
•	All the models are using the same pre-processed data for consistency and comparison.
DataPreprocessing.py 
All the pre-processing of data happens here.
•	Converting to lower case
•	Removing special characters
•	Stemming using Porter Stemmer
•	Removing all stop words in English except the word “not”
•	Finally dropping any NA records
NLTK.py 
This uses 6 Classifiers from Python’s Sci-kit Learn libraries called NLTK. The 6 classifiers used, and their corresponding accuracies are…
•	Naive Bayes Classifier – Overall Accuracy – 85.33
•	Multinomial Naive Bayes Classifier– Overall Accuracy – 85.68
•	NLTK Bernoulli Naive Bayes Classifier– Overall Accuracy – 84.60
•	Logistic Regression Classifier– Overall Accuracy – 83.51
•	Linear Support Vector Classifier – Overall Accuracy – 83.51
•	Stochastic Gradient Descent Classifier – Overall Accuracy – 84.50
Finally, to get the final output I have written a Voting Classifier which will give a single output of the best accuracy classifier.
RNN.py 
RNN stands for Recurrent Neural Network. A recurrent neural network (RNN) is a type of artificial neural network which uses sequential data or time series data. These deep learning algorithms are commonly used for ordinal or temporal problems, such as language translation, natural language processing (NLP), speech recognition, and image captioning.
•	I have used TensorFlow 2.0 with Keras layers, bidirectional LSTMs
•	Wikipedia definition of LSTM is - Long short-term memory (LSTM) is an artificial neural network used in the fields of artificial intelligence and deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. Such a recurrent neural network (RNN) can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition speech recognition, machine translation, robot control, video games, and healthcare.
BERT.py 
BERT stands for Bidirectional Encoder Representations from Transformers. Here’s the specifications of what I have used…
•	I have used the pre-trained BERT model provided by TFDistilBertForSequenceClassification. 
•	After setting the necessary parameters, the model was compiled, fitted and re-trained on my dataset and saved
app.py 
This uses the Python Flask for creating an API endpoint for POST.

