# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:02:44 2018

@author: Swapnil Masurekar
"""
from __future__ import division, print_function
print("Importing Libraries....")
import re, math, sys, numpy as np, nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from difflib import SequenceMatcher
from diff_match_patch import diff_match_patch
from collections import Counter
from apiclient import errors
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import base64
from email.mime.text import MIMEText


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
#                                   Gmail API Functions 
####################################################################################################################################################        
## Gmail API Functions labels-----------------------------------------------------------------------------------------------------------------------------
def CreateLabel(service, user_id, label_object):
  """Creates a new label within user's mailbox, also prints Label ID.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    label_object: label to be added.

  Returns:
    Created Label.
  """
  try:
    label = service.users().labels().create(userId=user_id,
                                            body=label_object).execute()
    print (label['id'])
    return label
  except (errors.HttpError):
    print ('An error occurred in creating label' )


def MakeLabel(label_name, mlv='show', llv='labelShow'):
  """Create Label object.

  Args:
    label_name: The name of the Label.
    mlv: Message list visibility, show/hide.
    llv: Label list visibility, labelShow/labelHide.

  Returns:
    Created Label.
  """
  label = {'messageListVisibility': mlv,
           'name': label_name,
           'labelListVisibility': llv,
           'color':{
                   'backgroundColor':'#000000',
                   'textColor':'#434343'
                   }
               
        }
  return label

def ListLabels(service, user_id):
  """Get a list all labels in the user's mailbox.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.

  Returns:
    A list all Labels in the user's mailbox.
  """
  try:
    response = service.users().labels().list(userId=user_id).execute()
    labels = response['labels']
#    for label in labels:
#      print ('Label id: %s - Label name: %s' % (label['id'], label['name']))
    return labels
  except (errors.HttpError):
    print ('An error occurred in listing labels')

## Gmail API Functions messages-----------------------------------------------------------------------------------------------------------------------------

def ListMessagesMatchingQuery(service, user_id, query=''):                      # if query = '', it would list all messages 
  """List all Messages of the user's mailbox matching the query.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    query: String used to filter messages returned.
    Eg.- 'from:user@some_domain.com' for Messages from a particular sender.

  Returns:
    List of Messages that match the criteria of the query. Note that the
    returned list contains Message IDs, you must use get with the
    appropriate ID to get the details of a Message.
  """
  try:
    response = service.users().messages().list(userId=user_id,
                                               q=query).execute()
    messages = []
    if 'messages' in response:
      messages.extend(response['messages'])

    while 'nextPageToken' in response:
      page_token = response['nextPageToken']
      response = service.users().messages().list(userId=user_id, q=query,
                                         pageToken=page_token).execute()
      messages.extend(response['messages'])

    return messages
  except (errors.HttpError):
    print ('An error occurred list messages from query' )


def ListMessagesWithLabels(service, user_id, label_ids=[]):
  """List all Messages of the user's mailbox with label_ids applied.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    label_ids: Only return Messages with these labelIds applied.

  Returns:
    List of Messages that have all required Labels applied. Note that the
    returned list contains Message IDs, you must use get with the
    appropriate id to get the details of a Message.
  """
  try:
    response = service.users().messages().list(userId=user_id,
                                               labelIds=label_ids).execute()
    messages = []
    if 'messages' in response:
      messages.extend(response['messages'])

    while 'nextPageToken' in response:
      page_token = response['nextPageToken']
      response = service.users().messages().list(userId=user_id,
                                                 labelIds=label_ids,
                                                 pageToken=page_token).execute()
      messages.extend(response['messages'])

    return messages
  except (errors.HttpError):
    print ('An error occurred in listing messages from labels') 

def GetMessage(service, user_id, msg_id):
  """Get a Message body and subject with given ID.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    msg_id: The ID of the Message required.

  Returns:
    A Message.
  """
  try:
    message = service.users().messages().get(userId=user_id, id=msg_id).execute()
    
#    print ('Message snippet: %s' % message['snippet'],message['payload']['headers'][19]['value'],message['payload']['headers'][16]['value'])
    s=message['payload']['headers'][16]['value']
    s=s[s.find("<")+1:s.find(">")]
    message={'body':message['snippet'],'subject':message['payload']['headers'][19]['value'],'From':s}
    return message                                                             # returns dictionary message body and subject
  except (errors.HttpError):
    print ('An error occurred while getting message')

def ModifyMessage(service, user_id, msg_id, msg_labels):
  """Modify the Labels on the given Message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    msg_id: The id of the message required.
    msg_labels: The change in labels.

  Returns:
    Modified message, containing updated labelIds, id and threadId.
  """
  print("Modifying Label....")
  try:
    message = service.users().messages().modify(userId=user_id, id=msg_id,
                                                body=msg_labels).execute()

#    label_ids = message['labelIds']

#    print ('Message ID: %s - With Label IDs %s' % (msg_id, label_ids))
    return message
  except (errors.HttpError):
    print ('An error occurred while modifying message')

"""Send an email message from the user's account.
"""


def SendMessage(service, user_id, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  print("Sending message....")
  try:
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
#    print ('Message Id: %s' % message['id'])
    return message
  except (errors.HttpError):
    print ('An error occurred while sending message')


def CreateMessage(sender, to, subject, message_text):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

  Returns:
    An object containing a base64url encoded email object.
  """
  message = MIMEText(message_text)
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject
  return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


# User defined functions----------------------------------------------------------------------------------------------------------------------------------------------------
def get_label_id(label_name, service, user_id):
    labels = ListLabels(service, user_id)
    for label in labels:
        if(label['name']==label_name):
            return label['id']

def CreateMsgLabels():
  """Create object to update labels.
  """
  return {'removeLabelIds': [], 'addLabelIds': []}

def identify_latest_received_message(service, user_id, message_list):           # Returns id of latest received messsage
    for msg in message_list:
        message = service.users().messages().get(userId=user_id, id=msg['id']).execute()
        if(message['labelIds']!=['SENT']):
            return msg['id']

def argmax_list(lst):
  return lst.index(max(lst))
#----------------------------------------------------------------------------------------------------------------------------------------------------

'''
####################################################################################################################################################
#                                   Automated Reply Complete main code 
####################################################################################################################################################        
'''
# Setup the Gmail API service----------------------------------------------------------------------------------------------------------------------------------------------------
SCOPES = 'https://www.googleapis.com/auth/gmail.modify'
store = file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    print("here")
    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
    creds = tools.run_flow(flow, store)
service = build('gmail', 'v1', http=creds.authorize(Http()))
user_id="me"

# Get the latest Received Message----------------------------------------------------------------------------------------------------------------------------------------------------
message_list=ListMessagesMatchingQuery(service, user_id, query='')
latest_received_msg_id = identify_latest_received_message(service, user_id, message_list)
message_read= GetMessage(service, user_id, latest_received_msg_id)
print("Latest received message is: ","\n    Message subject: ",message_read['subject'],"\n    Message body is: ",message_read['body'],"\n    From: ",message_read['From'])

####################################################################################################################################################
#                                   Generating Reply for the message
####################################################################################################################################################        
print("Generating the reply for the message....")

test_input_og=message_read['body']


import warnings
warnings.filterwarnings('ignore') # To ignore UserWarnings and DeprecationWarning

# Reading the cleaned dataset-------------------------------------------------------------------------------------------------------------------------
print("Reading the dataset....")
filenames=["certificate_clean","battery_clean"]
category_name=["certificate","battery"]
corpus_piazza_stemmed=[]
query_class_og=[]
for i in range(len(filenames)):
    file=open(filenames[i]+".txt","r")
    for line in file:
        corpus_piazza_stemmed.append(line)
        query_class_og.append(category_name[i])

# Creating the Bag of Words model for classification using bayes theorem----------------------------------------------------------------------------------------------
corpus_piazza_classify=corpus_piazza_stemmed
query_class=query_class_og
print("Creating Bag of words model....")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 50)
corpus_piazza_classify = cv.fit_transform(corpus_piazza_classify).toarray()
    
# Training the model or loading the trained model-----------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_query_class = LabelEncoder()
query_class = labelencoder_query_class.fit_transform(query_class)

# Fitting Naive Bayes to the Dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(corpus_piazza_classify, query_class)

# Predicting the category in which the latest read mail belongs to....
test_input_clean = clean_dataset([test_input_og])
test_input = cv.transform(test_input_clean).toarray()
test_prediction=classifier.predict(test_input)
test_prediction_proba=classifier.predict_proba(test_input)                      # Get probability for each category for given test input

test_prediction_text=labelencoder_query_class.inverse_transform(test_prediction)
print("The query belong to category --> ",test_prediction_text[0])

# Setting the label to the mail using gmail API---------------------------------------------------------------------------------------------------------------------------------------------------------
# Modifying message's label
print("Adding label to the latest received message....")
msg_labels=CreateMsgLabels()                                                    # Create object to update labels
msg_labels['addLabelIds']=[get_label_id(test_prediction_text[0], service, user_id)]
ModifyMessage(service, user_id, latest_received_msg_id, msg_labels)

# Selecting the most appropriate reply by sentence similarity-------------------------------------------------------------------------------------------------------------------------------------------
## Check for sentence similarity in predicted category and suggest corresponding reply------------------------------------------------------------------------------------------------
similarity_matrix=[[],[]]
test_prediction_string=labelencoder_query_class.inverse_transform(test_prediction)
flag=0
for i in range(len(query_class_og)):
    if (query_class_og[i] == test_prediction_string[0]):
        print(i)
#        print (corpus_piazza_stemmed[i],corpus_piazza_text[i],test_input_og,test_input)
        similarity_matrix[0].append(similarity(corpus_piazza_stemmed[i],test_input_clean[0],False))
        similarity_matrix[1].append(similarity(corpus_piazza_stemmed[i],test_input_clean[0],True))
        if(similarity_matrix[0][len(similarity_matrix[0])-1] > 0.4 or similarity_matrix[1][len(similarity_matrix[1])-1] > 0.4):
            flag=1
            break
        
# Load all replies from predicted category
file=open(test_prediction_text[0]+"_reply.txt","r")
all_replies_from_predicted_category=[]
send_flag=1
for line in file:
    all_replies_from_predicted_category.append(line)


if(flag==1):
    generated_reply=all_replies_from_predicted_category[len(similarity_matrix[0])-1]
else:
    index1=argmax_list(similarity_matrix[0]) 
    index2=argmax_list(similarity_matrix[1])
    if(similarity_matrix[0][index1]<0.4 or similarity_matrix[1][index2]<0.4):
        print("NO REPLY GENERATED....")
        send_flag=0
    else:
        if((similarity_matrix[0][index1]+similarity_matrix[1][index1])>(similarity_matrix[0][index2]+similarity_matrix[1][index2])):
            generated_reply=all_replies_from_predicted_category[index1]
        else:
            generated_reply=all_replies_from_predicted_category[index2]


####################################################################################################################################################        

# Sending generated reply----------------------------------------------------------------------------------------------------------------------------------------------------
if(send_flag==1):
    print("Generated Reply is: ",generated_reply)
    print("Sending Reply....")
    message_text=generated_reply
    message_send = CreateMessage(user_id, message_read['From'], 'Re:'+message_read['subject'], message_text)                    #'Re:'+message_read['subject'] fo replying
    SendMessage(service, user_id, message_send)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Program ended succesfully")