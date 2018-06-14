# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:12:07 2018

@author: Swapnil Masurekar
"""
from __future__ import division
import re, math, sys, numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from difflib import SequenceMatcher
from diff_match_patch import diff_match_patch
from collections import Counter


# GLobal Variables ---------------------------------------------------------------------------------------------------------------------------------
# Parameters to the algorithm. Currently set to values that was reported in the paper to produce best results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0
# User defined functions ---------------------------------------------------------------------------------------------------------------------------

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

####################################################################################################################################################
#                                   Sentence similarity functions 
####################################################################################################################################################        
## word similarity ---------------------------------------------------------------------------------------------------------------------------------

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim is not None:
#                   print("here")
                   if sim > max_sim:
                       max_sim = sim
                       best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxsize#maxint
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize#maxint
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
#                if hypernyms_1.has_key(lcs_candidate):
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
#                if hypernyms_2.has_key(lcs_candidate):
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))

######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
#                if not brown_freqs.has_key(word):
                if not word in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
#    n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    n = 0 if not lookup_word in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are  further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
        (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)
        
def compute_similarity(a,b):                                                   # uisng python native library difflib 
    return SequenceMatcher(None, a, b).ratio()                                  # Return similarity in terms of probability

def compute_similarity_and_diff(text1, text2):                                  # using diff_match_patch
# Finding degree of similiarity between 2 sentences
    dmp = diff_match_patch()
    dmp.Diff_Timeout = 0.0
    diff = dmp.diff_main(text1, text2, False)
    # similarity
    common_text = sum([len(txt) for op, txt in diff if op == 0])
    text_length = max(len(text1), len(text2))
    sim = common_text / text_length
    return sim, diff   

def compute_similarity_jellyfish(a,b):
    return
    

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)


####################################################################################################################################################
    
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
corpus_piazza_text=[]
query_class_og=[]
for filename in filenames:
    file=open(filename+".txt","r")
    for line in file:
        corpus_piazza_text.append(line)
        query_class_og.append(filename)
    
corpus_piazza_stemmed=clean_dataset(corpus_piazza_text)                         # corpus_piazza_text: Contains raw sentences
                                                                                # corpus_piazza: Contains stemmed sentences

corpus_piazza_rnn=corpus_piazza_stemmed
# Tokenizing using keras------------------------------------------------------------------------------------------------------
from keras.preprocessing.text import Tokenizer
# We create a tokenizer, configured to only take into account the top most common words
tokenizer_piazza = Tokenizer(num_words=100)

# This builds the word index, fitting tokenizer on clean dataset
tokenizer_piazza.fit_on_texts(corpus_piazza_rnn)

# This turns strings into lists of integer indices.
corpus_piazza_rnn = tokenizer_piazza.texts_to_sequences(corpus_piazza_rnn)

# You could also directly get the one-hot binary representations.
# Note that other vectorization modes than one-hot encoding are supported!
corpus_piazza_rnn = tokenizer_piazza.texts_to_matrix(corpus_piazza_rnn, mode='binary')

# This is how you can recover the word index that was computed
#word_index = tokenizer_piazza.word_index
#print('Found %s unique tokens.' % len(word_index))                             eg: Found 100 unique tokens

corpus_piazza_classify=corpus_piazza_stemmed
# Shuffling the dataset----------------------------------------------------------------------------------------------------------------------------------------
query_class=query_class_og
#from sklearn.utils import shuffle
#corpus_piazza_classify,query_class=shuffle(corpus_piazza_classify,query_class) # corpus_piazza_classify,query_class is shuffled

# Creating the Bag of Words model for classification using bayes theorem----------------------------------------------------------------------------------------------
print("Creating Bag of words model....")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50)
corpus_piazza_classify = cv.fit_transform(corpus_piazza_classify).toarray()

from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_query_class = LabelEncoder()
query_class = labelencoder_query_class.fit_transform(query_class)

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#corpus_piazza_classify_train, corpus_piazza_classify_test, query_class_train, query_class_test = train_test_split(corpus_piazza_classify, query_class, test_size = 0.05, random_state = 0)

# Fitting Naive Bayes to the Dataset----------------------------------------------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(corpus_piazza_classify, query_class)

# Predicting the Test set results
#query_class_pred = classifier.predict(corpus_piazza_test)

# Model evaluation---------------------------------------------------------------------------------------------------------------------------------------------
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(query_class_test, query_class_pred)

# Applying k-Fold Cross Validation
print("Running k-fold crosss-validation....")
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = corpus_piazza_classify, y = query_class, cv = 10)
print("Mean accuracy is: ",accuracies.mean())
print("Standard Deviation is: ",accuracies.std())

# Taking test input--------------------------------------------------------------------------------------------------------------------------------------------
test_input_og=[]
test_input_og.append(input("Enter string: "))
#test_input=["We haven't received our certificate"]
test_input=clean_dataset(test_input_og)
test_input = cv.transform(test_input).toarray()
test_prediction=classifier.predict(test_input)
test_prediction_proba=classifier.predict_proba(test_input)                      # Get probability for each category for given test input
print("The query belong to category --> ",labelencoder_query_class.inverse_transform(test_prediction))

## Check for sentence similarity in predicted category and suggest corresponding reply------------------------------------------------------------------------------------------------
similarity_matrix=[[],[],[]]
test_prediction_string=labelencoder_query_class.inverse_transform(test_prediction)
test_input=clean_dataset(test_input_og)
for i in range(len(query_class_og)):
    if (query_class_og[i] == test_prediction_string[0]):
        print(i)
#        print (corpus_piazza_stemmed[i],corpus_piazza_text[i],test_input_og,test_input)
        similarity_matrix[0].append(similarity(corpus_piazza_stemmed[i],test_input[0],False))
        similarity_matrix[1].append(similarity(corpus_piazza_stemmed[i],test_input[0],True))
        similarity_matrix[2].append(corpus_piazza_text[i])
    


# Similarity testing ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#sentence_pairs = [
#    ["When would i get my certificate ?", " When would we receive the certificate ?", 0],
#    ["When would i get my certificate ?", "Am i eligible for the certificate ?", 0],
#    ["I have completed the tasks, Am i eligible for certification ? ", "Am i eligible for the certificate ?", 0],
#    ["When would we get the battery ? ", "Battery not working", 0],
#    ["Power Bank is not functioning", "Battery not working", 0],
#]
#for sent_pair in sentence_pairs:
#    print("%s\t%s\t%.3f\t%.3f\t%.3f" % (sent_pair[0], sent_pair[1], sent_pair[2], 
#        similarity(sent_pair[0], sent_pair[1], False),
#        similarity(sent_pair[0], sent_pair[1], True)))
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Program ended succesfully")