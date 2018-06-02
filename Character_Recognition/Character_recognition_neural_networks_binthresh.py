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
from update_local_dataset_binthresh import *
#---------------------------------------------------------------------------------------------------------------------------------------------------
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
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur, applying gaussian blur to bitmap here


        
# filter image from grayscale to black and white--------------------------------------------------------------------------------------------------------
#        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
#                                          255,                                  # make pixels that pass the threshold full white
#                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian 
#                                          cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
#                                          11,                                   # size of a pixel neighborhood used to calculate threshold value
#                                          2)                                    # constant subtracted from the mean or weighted mean

        ret,imgThresh = cv2.threshold(imgBlurred,127,255,cv2.THRESH_BINARY_INV)

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

        npaROIResized_list=[]
        
        #----------------------------------------------------------------------
#        BLACK_IMAGE_WIDTH_HEIGHT=46
#        imgblack=cv2.imread("black.png") # reading black image for black background
#        imgblack = cv2.cvtColor(imgblack, cv2.COLOR_BGR2GRAY)
#        imgblack=imgblack[0:BLACK_IMAGE_WIDTH_HEIGHT,0:BLACK_IMAGE_WIDTH_HEIGHT] # cropping black image
        #----------------------------------------------------------------------
        
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
            ret,imgROIResized = cv2.threshold(imgROIResized,127,255,cv2.THRESH_BINARY_INV)      # Again threshold the image as resizing affects threshold effects
            # Padding black pixels in background--------------------------------------------------------------------------------------------------------
#            imgblack[int(BLACK_IMAGE_WIDTH_HEIGHT/2 - RESIZED_IMAGE_HEIGHT/2):int(BLACK_IMAGE_WIDTH_HEIGHT/2 + RESIZED_IMAGE_HEIGHT/2),
#                     int(BLACK_IMAGE_WIDTH_HEIGHT/2 - RESIZED_IMAGE_WIDTH/2):int(BLACK_IMAGE_WIDTH_HEIGHT/2 + RESIZED_IMAGE_WIDTH/2)]=imgROIResized
#            imgROIResized=cv2.resize(imgblack, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            #-------------------------------------------------------------------------------------------------------------------------------------------
            
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

#            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats
            
            for i in range(len(npaROIResized[0])):
                if npaROIResized[0][i]>128:
                    npaROIResized[0][i]=255
                else:
                    npaROIResized[0][i]=0
            

            npaROIResized_list.append(npaROIResized)
        return npaROIResized_list,imgTestingNumbers
    

####################################################################################################################################################
#                                   Character Recognition main code 
####################################################################################################################################################        

filename='Neural_networks_model_binthresh.h5'

case=input("Do you want to retrain the model?[y/n]")
if(case=='y' or case=='Y'):
    ##Importing datasets -----------------------------------------------------------------------------------------------------------------------------------
    
    # Getting local classification dataset------------------------------------------------------------------------------------------------------------------
    try:
        npaClassifications = np.loadtxt("classifications_binthresh.txt", np.float32)                  # read in training classifications
    except:
        print ("error, unable to open classifications_binthresh.txt, exiting program\n")
        os.system("pause")
    try:
        npaFlattenedImages = np.loadtxt("flattened_images_binthresh.txt", np.float32)                 # read in training images
    except:
        print ("error, unable to open flattened_images_binthresh.txt, exiting program\n")
        os.system("pause")
    
    # For testing purposes, for validation on sklearns digits dataset---------------------------------------------------------------------------------------
    #from sklearn import datasets
    #digits=datasets.load_digits()
    #npaFlattenedImages=digits['data']
    #npaClassifications=digits['target']
    
    # Loading mnist training dataset from train.csv---------------------------------------------------------------------------------------------------------
#    print("Reading mnist digits dataset....")
#    dataset = pd.read_csv('train.csv') # This dataset is not contour bounded edge to edge
#    npaFlattenedImages = dataset.iloc[:, 1:].values
#    npaClassifications = dataset.iloc[:, 0].values
#    # Thresholding dataset
#    print("Thresholding Dataset")
#    for i in range(len(npaFlattenedImages)):
#        for j in range(len(npaFlattenedImages[0])):
#            if(npaFlattenedImages[i][j]>90):
#                npaFlattenedImages[i][j]=255
#            else:
#                npaFlattenedImages[i][j]=0
    
    ####################################################################################################################################################
     

    
    # Feature Scaling for Flattened Images----------------------------------------------------------------------------------------------------------
    # Required for English Handwriting dataset 
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
    
    # Splitting the dataset into the Training set and Test set--------------------------------------------------------------------------------------
    #from sklearn.model_selection import train_test_split
    #npaFlattenedImages, npaFlattenedImages_test, npaClassifications, npaClassifications_test = train_test_split(npaFlattenedImages,
    #                                                                                                            npaClassifications,
    #                                                                                                            test_size = 0.15, 
    #                                                                                                            random_state = 0)
    
    
    # Initialising CLASSIFIERS for artificial neural networks---------------------------------------------------------------------------------------
    '''
    --> Neural Network classifier initialization with 1 input layer, 1 output layer and 2 hidden layers
    --> Neurons in hidden layer has uniform weights initialization and rectifier activation function associated
    --> Neuron in ouput layer has sigmoid activation fuction, softmax used here since NN as more than 2 outcomes
    '''
    import warnings
    warnings.filterwarnings('ignore') # To ignore UserWarnings and DeprecationWarning

    classifier_neural = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier_neural.add(Dense(output_dim = int((len(npaFlattenedImages[0])+len(npaClassifications[0]))/2.5), init = 'uniform', activation = 'relu', input_dim = int(len(npaFlattenedImages[0])))) # Using rectifier activation function
    
    # Adding the second hidden layer
    classifier_neural.add(Dense(output_dim = int((len(npaFlattenedImages[0])+len(npaClassifications[0]))/2.9), init = 'uniform', activation = 'relu')) # Using rectifier activation function for 2nd hidden layer
    
    # Adding the output layer
    classifier_neural.add(Dense(output_dim = len(npaClassifications[0]), init = 'uniform', activation = 'softmax')) # activation = 'softmax' --> in case of in case of output with more than two outcomes
    
    # Compiling the ANN
    classifier_neural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # loss = 'categorical_crossentropy' --> in case of output with more than two outcomes
    
    # Shuffling the Dataset, since validation split while fitting doesn't randomize the test set
    from sklearn.utils import shuffle
    npaFlattenedImages,npaClassifications=shuffle(npaFlattenedImages,npaClassifications)
    
    # Fitting the ANN to the Training set
    neural_networks_fitting_history=classifier_neural.fit(npaFlattenedImages, npaClassifications,
                                                          validation_split = 0.08,                                   # here in validation_split the split is not random it always take last 10% of the data, hence shuffle before fitting
                                                          batch_size = int(len(npaFlattenedImages[0])/30),
                                                          nb_epoch = 20)
                                                                    
    
    print("\nA record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values: ",neural_networks_fitting_history)

    classifier_neural.save(filename)
    
# Model Evaluation (need to split the dataset in training set and test set), like confusion matrix in sklearn model evaluation
#loss_and_acc = classifier_neural.evaluate(npaFlattenedImages_test, npaClassifications_test, batch_size=int(len(npaFlattenedImages_test[0])/20))
#print("AFter evaluation, Loss is: ",loss_and_acc[0],", Accuracy is: ",loss_and_acc[1])
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import load_model
classifier_neural = load_model(filename)

for test_image_number in range(5,6):
    test_image="test"+str(test_image_number)+".png"
    npaROIResized_list, imgTestingNumbers = image_feature_extraction.get_X_features_by_character_cropping("test_images/"+test_image) ## Get cropped characters ##
    
    ## Initialize empty strings for storing results from classification
    neural_network_results=""
    
    ## Predict classification results-----------------------------------------------------------------------------------------------------------------------
    for npaROIResized in npaROIResized_list:
        npaROIResized=sc_npaFlattenedImages.transform(npaROIResized)
#        print(npaROIResized)
        y_pred = classifier_neural.predict(npaROIResized)
        y_pred_inverted = labelencoder_npaClassifications.inverse_transform([np.argmax(y_pred[0, :])])
        neural_network_results=neural_network_results+chr(int(y_pred_inverted))
    
    ## Print Results ---------------------------------------------------------------------------------------------------------------------------------------
    print ("\nResults from Neural Networks: ====> " + neural_network_results + "\n")                 # show the full string
    
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press
    
    cv2.destroyAllWindows()             # remove windows from memory

    # Update local dataset if any character is incorrectly recognised--------------------------------------------------------------------------------------
    case=input("In case the of incorrect detection or for better learning, update the dataset for test"+str(test_image_number)+".png:\n Do you want to update the dataset? [y/n]")
    if(case=='y' or case=='Y'):
        # For updating pandas dataframe remember df = df.append({'A': i}, ignore_index=True) 
        update_data_binthresh("test_images/"+test_image)

print("Program ended successfully !!")
#---------------------------------------------------------------------------------------------------------------------------------------------------------
