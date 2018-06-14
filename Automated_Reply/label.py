# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:27:47 2018

@author: Swapnil Masurekar
"""
from __future__ import print_function
from apiclient import errors
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

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

    print ('Message snippet: %s' % message['snippet'],message['payload']['headers'][19]['value'])
    
    message={'body':message['snippet'],'subject':message['payload']['headers'][19]['value']}
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
  try:
    message = service.users().messages().modify(userId=user_id, id=msg_id,
                                                body=msg_labels).execute()

    label_ids = message['labelIds']

    print ('Message ID: %s - With Label IDs %s' % (msg_id, label_ids))
    return message
  except (errors.HttpError):
    print ('An error occurred while modifying message')



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
#----------------------------------------------------------------------------------------------------------------------------------------------------

####################################################################################################################################################
#                                                           Labelling main Code 
#################################################################################################################################################### 

    
# Setup the Gmail API service
SCOPES = 'https://www.googleapis.com/auth/gmail.modify'
store = file.Storage('credentials.json')
creds = store.get()
if not creds or creds.invalid:
    print("here")
    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
    creds = tools.run_flow(flow, store)
service = build('gmail', 'v1', http=creds.authorize(Http()))



# Make Label Object
label_object = MakeLabel("battery", mlv='show', llv='labelShow')

# Creating new gmail label
user_id="me"
#CreateLabel(service, user_id, label_object)


print(get_label_id('certificate', service, user_id))
message_list=ListMessagesMatchingQuery(service, user_id, query='')
message= GetMessage(service, user_id, message_list[0]['id'])
print("Message subject is:",message['subject'],", Message body is:",message['body'])

# Modifying message label
msg_labels=CreateMsgLabels()                                                    # Create object to update labels
msg_labels['removeLabelIds']=[get_label_id('certificate', service, user_id)]
ModifyMessage(service, user_id, message_list[0]['id'], msg_labels)