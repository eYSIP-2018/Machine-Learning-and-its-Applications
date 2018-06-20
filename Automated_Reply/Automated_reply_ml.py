# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:11:48 2018

@author: Swapnil Masurekar
"""
from __future__ import division, print_function
print("Importing Libraries....")
import re, numpy as np
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from apiclient import errors
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import base64
from email.mime.text import MIMEText
from sklearn.externals import joblib # to save the model for later use


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
# Reading the cleaned dataset-------------------------------------------------------------------------------------------------------------------------

test_input_og=message_read['body']

print("Reading Dataset....")
import pandas as pd
filename="Piazza_Dataset.txt"
Dataset=pd.read_csv(filename, sep='\t', lineterminator='\r')
corpus_piazza_queries=Dataset['QUERIES']
corpus_piazza_stemmed=clean_dataset(corpus_piazza_queries)
query_class_og=Dataset['CATEGORY'].tolist()
counter_categories=Counter(query_class_og)


all_words=[]        
for i in range(len(corpus_piazza_stemmed)):
    temp=corpus_piazza_stemmed[i].split()
    for j in range(len(temp)):
        all_words.append(temp[j])
        
# Python program to find the k most frequent words
# from data set

# Pass the split_it list to instance of Counter class.
Counter_all_words = Counter(all_words)
 # most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter_all_words.most_common(len(all_words))


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
category_name=[]
counter_category=Counter(query_class)
most_occur_category=counter_category.most_common(len(query_class))
for i in most_occur_category:
    category_name.append(i[0])
all_categories=clean_dataset(category_name)

corpus_piazza_classify_w = corpus_piazza_classify
manual_weight=2                                                                 # As corpus_piazza_classify is int64
for i in range(len(vocab)):
    if(vocab[i] in all_categories):
        for j in range(len(corpus_piazza_classify)):
            corpus_piazza_classify_w[j][i]=corpus_piazza_classify_w[j][i]*manual_weight
            
# Train the model---------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_query_class = LabelEncoder()
query_class = labelencoder_query_class.fit_transform(query_class)


# Applying Principle Component Analysis PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 2) # None
#corpus_piazza_classify = pca.fit_transform(corpus_piazza_classify)
#explained_variance = pca.explained_variance_ratio_
#
#from matplotlib import pyplot
#corpus_piazza_classify_temp1=[]
#corpus_piazza_classify_temp2=[]
#corpus_piazza_classify_temp3=[]
#for i in range(len(query_class_og)):
#    if(query_class_og[i] == 'certificate'):
#        corpus_piazza_classify_temp1.append(corpus_piazza_classify[i])
#    if(query_class_og[i] == 'battery'):
#        corpus_piazza_classify_temp2.append(corpus_piazza_classify[i])
#    if(query_class_og[i] == 'others'):
#        corpus_piazza_classify_temp3.append(corpus_piazza_classify[i])
#
#corpus_piazza_classify_temp1=np.array(corpus_piazza_classify_temp1) 
#corpus_piazza_classify_temp2=np.array(corpus_piazza_classify_temp2)
#corpus_piazza_classify_temp3=np.array(corpus_piazza_classify_temp3)
#       
#pyplot.scatter(corpus_piazza_classify_temp1[:, 0], corpus_piazza_classify_temp1[:, 1])
#pyplot.scatter(corpus_piazza_classify_temp2[:, 0], corpus_piazza_classify_temp2[:, 1])
#pyplot.scatter(corpus_piazza_classify_temp3[:, 0], corpus_piazza_classify_temp3[:, 1])



# Fitting Naive Bayes to the Dataset
from sklearn.naive_bayes import GaussianNB
classifier_upper_NB = GaussianNB()
classifier_upper_NB.fit(corpus_piazza_classify_w, query_class)

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
    classifier_category = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier_category.fit(train_set_category, train_y_category)
    filename_category='classifier_KNN_'+category+'.sav'
    joblib.dump(classifier_category, filename_category)


# Test the model---------------------------------------------------------------------------------------------------------------------------------------------------

# Prediction
test_input_clean = clean_dataset([test_input_og])
test_input = cv.transform(test_input_clean).toarray()
test_prediction = classifier_upper_NB.predict(test_input)
test_prediction_proba = classifier_upper_NB.predict_proba(test_input)                      # Get probability for each category for given test input
test_prediction_text=labelencoder_query_class.inverse_transform(test_prediction)
print("The query belong to category --> ",test_prediction_text[0])

# Load KNN model for predicted category----------------------------------------
filename_category='classifier_KNN_'+test_prediction_text[0]+'.sav'
classifier_category = joblib.load(filename_category) # Logistic regression model for multi-class classification loaded


send_flag=0
if(test_prediction_text[0]!="others"):
    # Load all replies from predicted category
    all_replies_from_predicted_category=[]
    send_flag=1
    for i in range(len(query_class_og)):
        if(test_prediction_text[0]==query_class_og[i]):
            all_replies_from_predicted_category.append(Dataset['REPLY'][i])
    
    pred=classifier_category.predict(test_input)
    reply_prediction_probability = classifier_category.predict_proba(test_input)
    generated_reply = all_replies_from_predicted_category[pred[0]]
    

####################################################################################################################################################        

# Setting the label to the mail using gmail API---------------------------------------------------------------------------------------------------------------------------------------------------------
# Modifying message's label
print("Adding label to the latest received message....")
msg_labels=CreateMsgLabels()                                                    # Create object to update labels
msg_labels['addLabelIds']=[get_label_id(test_prediction_text[0], service, user_id)]
ModifyMessage(service, user_id, latest_received_msg_id, msg_labels)

# Sending generated reply----------------------------------------------------------------------------------------------------------------------------------------------------
if(send_flag==1):
    print("Generated Reply is: ",generated_reply)
    print("Sending Reply....")
    message_text=generated_reply
    message_send = CreateMessage(user_id, message_read['From'], 'Re:'+message_read['subject'], message_text)                    #'Re:'+message_read['subject'] fo replying
    SendMessage(service, user_id, message_send)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Program ended succesfully")