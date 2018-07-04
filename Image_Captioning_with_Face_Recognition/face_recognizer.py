# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:38:30 2018

@author: Swapnil Masurekar
"""
import cv2
import os
from nltk.corpus import stopwords

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord



def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (150, 150, 0), 8)

def collect_labels():
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
    return labels_dic

def detect_hotwords(input_string):
    list_words =['man','men']
    string_words = input_string.split()                                         # tokenization
    hotwords_present = False
    hotwords =[]
    positions =[]
    for i, word in enumerate(string_words):
        if(word in list_words):
            hotwords_present = True
            positions.append(i)
            hotwords.append(word)
    return hotwords_present, hotwords, positions

def substitute_name(input_string, positions, predicted_name): 
    string_words =  input_string.split() 
    if(positions[0] != 0):
        if(string_words[positions[0]-1] in set(stopwords.words('english'))):
            del string_words[positions[0]-1]
            string_words[positions[0]-1] = predicted_name
        else:
             string_words[positions[0]] = predicted_name   
    else:
        string_words[positions[0]] = predicted_name
    
    input_string = ""
    for i in range(len(string_words)):
        input_string = input_string + " " +string_words[i]                      # removing eeee
     
    return input_string
    
    
    
"""
# MAIN FUNCTION--------------------------------------------------------------------------------------------------------------------------------------
"""
def generate_caption_on_face(input_string, img_path):
    # Detect if hotwords are present
    hotwords_present, hotwords, positions = detect_hotwords(input_string)                            # returns boolean
    
    if(hotwords_present):
        # Detect face in image
        detector = FaceDetector("xml/frontal_face.xml")                         # load face detector model object
                                                                                # load Eigen-face recognizer model object
        
        frame = cv2.imread(img_path)
        faces_coord = detector.detect(frame)                                    # detect face
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord)                         # norm pipeline
            for i, face in enumerate(faces):
                rec_eig = cv2.face.EigenFaceRecognizer_create()
                rec_eig.read("rec_eig.yml")
                pred,conf = rec_eig.predict(face)                               # get prediction and confidence
                labels_dic = collect_labels()
                predicted_name = labels_dic[pred].capitalize()                  # Recognize person
        # Remove eeee
        string_words =  input_string.split() 
        input_string = ""
        for i in range(len(string_words) -1 ):
            input_string = input_string + " " +string_words[i]                      # removing eeee
        
        
        # Substitute hot-words by person's name
        if(positions != [] and len(faces_coord) != 0):
            input_string = substitute_name(input_string, positions, predicted_name)
        
    return input_string
        
