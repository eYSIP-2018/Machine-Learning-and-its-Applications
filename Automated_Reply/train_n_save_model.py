# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:04:42 2018

@author: Swapnil Masurekar
"""
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib # to save the model for later use

filename_upper_NB='classifier_upper_NB.sav'

def clean_dataset(Dataset):
# Cleaning the dataset: Removing punctuations and stopwords
    corpus = []
    for i in range(0, len(Dataset)):                                                # Creating corpus
        review = re.sub('[^a-zA-Z?]', ' ', Dataset[i])                              # Removing punctuations
        review = review.lower()                                                     # Lower-case
        review = review.split()                                                     # Splitting sentence into list of words (tokenization)
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)                                                   # Joining the words again
        corpus.append(review)
    return corpus

# Reading the cleaned dataset-------------------------------------------------------------------------------------------------------------------------
print("Reading the dataset....")
filenames=["certificate_clean","battery_clean","others_clean"]
category_name=["certificate","battery","others"]
corpus_piazza_stemmed=[]
query_class_og=[]
for i in range(len(filenames)):
    file=open(filenames[i]+".txt","r")
    for line in file:
        corpus_piazza_stemmed.append(line)
        query_class_og.append(category_name[i])

# Loading the replies


all_words=[]        
for i in range(len(corpus_piazza_stemmed)):
    temp=corpus_piazza_stemmed[i].split()
    for j in range(len(temp)):
        all_words.append(temp[j])
        
# Python program to find the k most frequent words
# from data set
from collections import Counter
# Pass the split_it list to instance of Counter class.
Counter_all_words = Counter(all_words)
 # most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter_all_words.most_common(1500)


# Creating Vocabulary
vocab=[]
for i in range(len(most_occur)):
    vocab.append(most_occur[i][0])
    
# Creating the Bag of Words model for classification using bayes theorem----------------------------------------------------------------------------------------------
corpus_piazza_classify=corpus_piazza_stemmed
query_class=query_class_og
print("Creating Bag of words model....")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(vocabulary=vocab)
corpus_piazza_classify = cv.fit_transform(corpus_piazza_classify).toarray()

# Giving weights to words which belong to category--------------------------------------------------------------------------------------------------------------------
all_categories=[]
counter_category=Counter(query_class)
most_occur_category=counter_category.most_common(len(query_class))
for i in most_occur_category:
    all_categories.append(i[0])
all_categories=clean_dataset(all_categories)

manual_weight=2                                                                 # As corpus_piazza_classify is int64
for i in range(len(vocab)):
    if(vocab[i] in all_categories):
        for j in range(len(corpus_piazza_classify)):
            corpus_piazza_classify[j][i]=corpus_piazza_classify[j][i]*manual_weight
            
# Train the model---------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_query_class = LabelEncoder()
query_class = labelencoder_query_class.fit_transform(query_class)

# Fitting Naive Bayes to the Dataset
from sklearn.naive_bayes import GaussianNB
classifier_upper_NB = GaussianNB()
classifier_upper_NB.fit(corpus_piazza_classify, query_class)

# Training KNN n-inputs and n-outputs for similarity purpose-------------------

# Training KNN for all respective categories and storing in file
# eg: filename= classifier_KNN_certificate, classifier_KNN_battery, ....
for category in category_name:
    train_set_category=[]
    for i in range(len(corpus_piazza_classify)):
        if(query_class_og[i]==category):    
            train_set_category.append(corpus_piazza_classify[i])
    train_y_category=range(len(train_set_category))
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier_category = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
    classifier_category.fit(train_set_category, train_y_category)
    filename_category='classifier_KNN_'+category+'.sav'
    joblib.dump(classifier_category, filename_category)

# Save the model---------------------------------------------------------------------------------------------------------------------------------------------------
joblib.dump(classifier_upper_NB, filename_upper_NB)                                     # save model in filename

# Test the model---------------------------------------------------------------------------------------------------------------------------------------------------

# Load the model
classifier_upper_NB = joblib.load(filename_upper_NB) # Logistic regression model for multi-class classification loaded

# Prediction
test_input_og=[]
test_input_og.append(input("Enter string: "))
#test_input=["We haven't received our certificate"]
test_input=clean_dataset(test_input_og)
test_input = cv.transform(test_input).toarray()
test_prediction=classifier_upper_NB.predict(test_input)
test_prediction_proba=classifier_upper_NB.predict_proba(test_input)                      # Get probability for each category for given test input
test_prediction_text=labelencoder_query_class.inverse_transform(test_prediction)
print("The query belong to category --> ",test_prediction_text[0])

# Load KNN model for predicted category----------------------------------------
filename_category='classifier_KNN_'+test_prediction_text[0]+'.sav'
classifier_category = joblib.load(filename_category) # Logistic regression model for multi-class classification loaded

if(test_prediction_text[0]!="others"):
    # Load all replies from predicted category
    file=open(test_prediction_text[0]+"_reply.txt","r")
    all_replies_from_predicted_category=[]
    send_flag=1
    for line in file:
        all_replies_from_predicted_category.append(line)
    
    pred=classifier_category.predict(test_input)
    print("Predicted Reply is: ", all_replies_from_predicted_category[pred[0]], pred)