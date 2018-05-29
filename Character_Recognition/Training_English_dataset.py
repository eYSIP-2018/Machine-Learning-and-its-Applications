# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:50:42 2018

@author: Swapnil Masurekar
"""
import sys
import numpy as np
import cv2
import os

def decode_character(training_folder_number):
    decoding_array=[]
    for i in range(48,58):
        decoding_array.append(chr(i))
    for i in range(65,91):
        decoding_array.append(chr(i))
    for i in range(97,123):
        decoding_array.append(chr(i))
    return decoding_array[training_folder_number-1]

def decode_path(training_folder_number, training_image_number):
    if(training_folder_number<10 and training_image_number<10 ):
        path="Dataset/English/Hnd/Img/Sample00"+str(training_folder_number)+"/img00"+str(training_folder_number)+"-00"+str(training_image_number)+".png"
    elif(training_folder_number<10 and training_image_number>=10 ):
        path="Dataset/English/Hnd/Img/Sample00"+str(training_folder_number)+"/img00"+str(training_folder_number)+"-0"+str(training_image_number)+".png"
    elif(training_folder_number>=10 and training_image_number<10 ):
        path="Dataset/English/Hnd/Img/Sample0"+str(training_folder_number)+"/img0"+str(training_folder_number)+"-00"+str(training_image_number)+".png"
    else:
        path="Dataset/English/Hnd/Img/Sample0"+str(training_folder_number)+"/img0"+str(training_folder_number)+"-0"+str(training_image_number)+".png"
    return path
# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

######################################## Main Code ###########################################################


for training_folder_number in range(1,37):
    
    for training_image_number in range(1,56):
        print("Traning folder Sample0"+str(training_folder_number),"Traning image0"+str(training_image_number))
        imgTrainingNumbers = cv2.imread(decode_path(training_folder_number, training_image_number))            # read in training numbers image
        
        if imgTrainingNumbers is None:                          # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
                                                          # and exit function (which exits program)
        
        
        imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur
        
                                                        # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                          255,                                  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                          11,                                   # size of a pixel neighborhood used to calculate threshold value
                                          2)                                    # constant subtracted from the mean or weighted mean
        
        #    cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference
        
        
        imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
        
        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points
        
                                    # declare empty numpy array, we will use this to write to file later
                                    # zero rows, enough cols to hold all image data
        npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        
        intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
        
        
        for npaContour in npaContours:                          # for each contour
            if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect
        
                                                    # draw rectangle around each contour as we ask user for input
                cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                              (intX, intY),                 # upper left corner
                              (intX+intW,intY+intH),        # lower right corner
                              (0, 0, 255),                  # red
                              2)                            # thickness
        
                imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage
        
    #            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
    #            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
    #            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it
    #    
    #            intChar = cv2.waitKey(0)                     # get key press
    #    
    #            if intChar == 27:                   # if esc key was pressed
    #                sys.exit()                      # exit program
    #            cv2.destroyAllWindows()             # remove windows from memory
                intClassifications.append(ord(decode_character(training_folder_number)))                                                # append classification char to integer list of chars (we will convert to float later before writing to file)
        
        npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
        npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
               
        
        fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats
        
        npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later
        
        
        
        try:
            npaClassifications_og = np.loadtxt("classifications_english.txt", np.float32)                  # read in training classifications
        except:
            print ("error, unable to open classifications_english.txt, exiting program\n")
            os.system("pause")
        
        try:
            npaFlattenedImages_og = np.loadtxt("flattened_images_english.txt", np.float32)                 # read in training images
        except:
            print ("error, unable to open flattened_images_english.txt, exiting program\n")
            os.system("pause")
        
        
        
        npaFlattenedImages_loadable= np.append(npaFlattenedImages_og,npaFlattenedImages,axis=0) # adding new images to load in text file
        npaClassifications_loadable = np.append(npaClassifications_og,npaClassifications)  # adding new classifications to load in text file
        
        np.savetxt("classifications_english.txt", npaClassifications_loadable)           # write flattened images to text file
        np.savetxt("flattened_images_english.txt", npaFlattenedImages_loadable)          # write classifiactions to text file

print("\nProgram executed Successfully !!")



###################################################################################################





