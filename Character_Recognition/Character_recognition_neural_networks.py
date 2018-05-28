# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:30:03 2018

@author: Swapnil Masurekar
"""



import cv2
import numpy as np
import operator
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd # to load the dataset
from update_local_dataset import *

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

####################################################################################################################################################
#                        Contour with data class with member variables to check contour validity and getting bounding rectange info 
#################################################################################################################################################### 
class ContourWithData():

# member variables 
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculate_Rect_Top_Left_Point_And_Width_And_Height(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def check_If_Contour_Is_Valid(self):                            
        if self.fltArea < MIN_CONTOUR_AREA: return False        
        return True
####################################################################################################################################################
#                                   Image feature extraction class 
#################################################################################################################################################### 
class image_feature_extraction():

    MIN_CONTOUR_AREA = 100

    RESIZED_IMAGE_WIDTH = 20
    RESIZED_IMAGE_HEIGHT = 30
    def get_X_features_by_character_cropping(input_img):
        '''
    	# Function: get_X_features_by_character_cropping
    	# Input: String -> input_img -> image name and path in format "test_images/test7.png" 
    	# Output: npaROIResized_list,imgTestingNumbers
    	# Logic: 1. Get imgTestingNumbers by reading the image from path input_img
                2. Filter image from grayscale to black and white
                3. Get information of each character in image by finding contours
                4. Crop out the ROI containg character then resize it to (20 x 30) and get resized images list
                5. Flatten the image to 1d numpy vector which be later used for prediction
    	# Example call: npaROIResized_list, imgTestingNumbers = image_feature_extraction.get_X_features_by_character_cropping("test_images/"+test_image)
        '''
        allContoursWithData = []                            # declare empty lists,
        validContoursWithData = []                          # we will fill these shortly
        imgTestingNumbers = cv2.imread(input_img)           # read in testing numbers image

        if imgTestingNumbers is None:                           # if image was not read successfully
            print ("error: image not read from file \n\n")      # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return                                              # and exit function (which exits program)


        imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

# filter image from grayscale to black and white--------------------------------------------------------------------------------------------------------
        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                          255,                                  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                          11,                                   # size of a pixel neighborhood used to calculate threshold value
                                          2)                                    # constant subtracted from the mean or weighted mean

        imgThreshCopy = imgThresh.copy()                                        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

# Get information of each character in image by finding contours----------------------------------------------------------------------------------------
        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

        
        for npaContour in npaContours:# for each contour
            contourWithData = ContourWithData()                                             # instantiate a contour with data object
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
            contourWithData.calculate_Rect_Top_Left_Point_And_Width_And_Height()            # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
            allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data


        for contourWithData in allContoursWithData:                 # for all contours
            if contourWithData.check_If_Contour_Is_Valid():         # check if valid depending on contour area
                validContoursWithData.append(contourWithData)       # if so, append to valid contour list

        validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

# 
        npaROIResized_list=[]

        for contourWithData in validContoursWithData:            # for each contour
                                                    # draw a green rect around the current char
            cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (0, 255, 0),              # green
                          2)                        # thickness

            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

            npaROIResized_list.append(npaROIResized)
        return npaROIResized_list,imgTestingNumbers
    

####################################################################################################################################################
#                                   Character Recognition main code 
####################################################################################################################################################        


##Importing datasets -----------------------------------------------------------------------------------------------------------------------------------

# Getting local classification dataset------------------------------------------------------------------------------------------------------------------
try:
    npaClassifications = np.loadtxt("classifications_english.txt", np.float32)                  # read in training classifications
except:
    print ("error, unable to open classifications_english.txt, exiting program\n")
    os.system("pause")


try:
    npaFlattenedImages = np.loadtxt("flattened_images_english.txt", np.float32)                 # read in training images
except:
    print ("error, unable to open flattened_images_english.txt, exiting program\n")
    os.system("pause")

# Feature Scaling for Flattened Images----------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc_npaFlattenedImages = StandardScaler()
npaFlattenedImages = sc_npaFlattenedImages.fit_transform(npaFlattenedImages)

# Encoding done for Classifications-------------------------------------------------------------------------------------------------------------
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_npaClassifications = LabelEncoder()                                    # Label encoding not needed in this case as classifications are ASCII numeric values
                                                                                    # But here we use label encoder to later use inverse transform method for decoding purposes
npaClassifications[:, 0] = labelencoder_npaClassifications.fit_transform(npaClassifications[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
npaClassifications = onehotencoder.fit_transform(npaClassifications).toarray()


# Initialising CLASSIFIERS for artificial neural networks---------------------------------------------------------------------------------------
'''
--> Neural Network classifier initialization with 1 input layer, 1 output layer and 2 hidden layers
--> Neurons in hidden layer has uniform weights initialization and rectifier activation function associated
--> Neuron in ouput layer has sigmoid activation fuction, softmax used here since NN as more than 2 outcomes
'''
classifier_neural = Sequential()

# Adding the input layer and the first hidden layer
classifier_neural.add(Dense(output_dim = int((len(npaFlattenedImages[0])+len(npaClassifications[0]))/2), init = 'uniform', activation = 'relu', input_dim = int(len(npaFlattenedImages[0])))) # Using rectifier activation function

# Adding the second hidden layer
classifier_neural.add(Dense(output_dim = int((len(npaFlattenedImages[0])+len(npaClassifications[0]))/2), init = 'uniform', activation = 'relu')) # Using rectifier activation function for 2nd hidden layer

# Adding the output layer
classifier_neural.add(Dense(output_dim = len(npaClassifications[0]), init = 'uniform', activation = 'softmax')) # activation = 'softmax' --> in case of in case of output with more than two outcomes

# Compiling the ANN
classifier_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # loss = 'categorical_crossentropy' --> in case of output with more than two outcomes

# Fitting the ANN to the Training set
classifier_neural.fit(npaFlattenedImages, npaClassifications, batch_size = 10, nb_epoch = 5)

#-----------------------------------------------------------------------------------------------------------------------------------------------

test_image="test11.png"
npaROIResized_list, imgTestingNumbers = image_feature_extraction.get_X_features_by_character_cropping("test_images/"+test_image) ## Get cropped characters ##

## Initialize empty strings for storing results from classification
neural_network_results=""

## Predict classification results-----------------------------------------------------------------------------------------------------------------------
for npaROIResized in npaROIResized_list:
    npaFlattenedImages=sc_npaFlattenedImages.transform(npaFlattenedImages)
    y_pred = classifier_neural.predict(npaROIResized)
    y_pred_inverted = labelencoder_npaClassifications.inverse_transform([np.argmax(y_pred[0, :])])
    neural_network_results=neural_network_results+chr(int(y_pred_inverted))

## Print Results ---------------------------------------------------------------------------------------------------------------------------------------
print ("\nResults from Neural Networks: " + neural_network_results + "\n")                 # show the full string

cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
cv2.waitKey(0)                                          # wait for user key press

cv2.destroyAllWindows()             # remove windows from memory

# Update local dataset if any character is incorrectly recognised--------------------------------------------------------------------------------------
case=input("In case the of incorrect detection or for better learning, update the dataset and train model again:\n Do you want to update the dataset? [y/n]")
if(case=='y' or case=='Y'):
    update_data("test_images/"+test_image)
print("Program ended successfully")