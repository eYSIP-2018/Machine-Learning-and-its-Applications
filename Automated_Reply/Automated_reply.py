# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:12:07 2018

@author: Swapnil Masurekar
"""

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_dataset(Dataset):
# Cleaning the dataset: Removing punctuations and stopwords
    corpus = []
    for i in range(0, len(Dataset)):                                                # Creating corpus
        review = re.sub('[^a-zA-Z?]', ' ', Dataset[i])                              # Removing punctuations
        review = review.lower()                                                     # Lower-case
        review = review.split()                                                     # Splitting sentence into list of words
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)                                                   # Joining the words again
        corpus.append(review)
    return corpus
    
# Load email corpus--------------------------------------------------------------------------------------------------------------

#corpus_filename='corpus_emails'
#with open (corpus_filename, 'rb') as fp:
#    corpus_emails = pickle.load(fp)                                             # Loading the stored corpus

####################################################################################################################################################
#                                   Automated Reply main code 
####################################################################################################################################################        

import warnings
warnings.filterwarnings('ignore') # To ignore UserWarnings and DeprecationWarning

print("Reading the dataset....")
filenames=["certificate","battery"]
corpus_piazza=[]
query_class=[]
for filename in filenames:
    file=open(filename+".txt","r")
    for line in file:
        corpus_piazza.append(line)
        query_class.append(filename)
    
corpus_piazza=clean_dataset(corpus_piazza)

# Shuffling the dataset
from sklearn.utils import shuffle
corpus_piazza,query_class=shuffle(corpus_piazza,query_class)

# Creating the Bag of Words model----------------------------------------------------------------------------------------------
print("Creating Bag of words model....")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50)
corpus_piazza = cv.fit_transform(corpus_piazza).toarray()

from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_query_class = LabelEncoder()
query_class = labelencoder_query_class.fit_transform(query_class)

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#corpus_piazza_train, corpus_piazza_test, query_class_train, query_class_test = train_test_split(corpus_piazza, query_class, test_size = 0.05, random_state = 0)

# Fitting Naive Bayes to the Dataset-------------------------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(corpus_piazza, query_class)

# Predicting the Test set results
#query_class_pred = classifier.predict(corpus_piazza_test)

# Model evaluation-------------------------------------------------------------------------------------------------------------
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(query_class_test, query_class_pred)

# Applying k-Fold Cross Validation
print("Running k-fold crosss-validation....")
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = corpus_piazza, y = query_class, cv = 10)
print("Mean accuracy is: ",accuracies.mean())
print("Standard Deviation is: ",accuracies.std())

# Taking test input------------------------------------------------------------------------------------------------------------
test_input=[]
test_input.append(input("Enter string: "))
#test_input=["We haven't received our certificate"]
test_input=clean_dataset(test_input)
test_input = cv.transform(test_input).toarray()
test_prediction=classifier.predict(test_input)
test_prediction_proba=classifier.predict_proba(test_input)                      # Get probability for each category for given test input
print("The query belong to category --> ",labelencoder_query_class.inverse_transform(test_prediction))
