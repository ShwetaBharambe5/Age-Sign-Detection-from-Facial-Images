import sys
sys.path.append("/content/drive/MyDrive/B7 project/AI-B7-Project/notebooks")
import base_model
import base_model_efficientNetB0

import cv2
import shutil
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join,exists
from os import mkdir
from matplotlib.pyplot import imshow,subplot,figure
from cv2 import imread,cvtColor,COLOR_BGR2RGB,rectangle

import getPatches

#################################################
#-----------------REQUIREMENTS------------------#
# 1. MAKE THE CROPPED_PATH_DIR FOLDER 
#     IN THE DATABASE (INSTANCE STORAGE)
# 2. ADD PATH OF THE IMAGE UPLOADED TO IMG_PATH
#-----------------------------------------------#
#################################################

IMG_PATH = '/content/test_images'  ##### Change it to the path where the uploaded image is stored from website
IMG_DIMS = (299,299,3)
CROPPED_PATCH_DIR = "/content/cropped_dir" #### *REQUIREMENT*
mkdir(IMG_PATH) if not exists(IMG_PATH) else None

#################################################
#-----------------------------------------------#
#################################################

def wrinkle_model_func(model_name):

  if model_name == 'inception':
    wrinkle_model = base_model.detection_model('wrinkle')
    wrinkle_model.load_weights(base_model.wrinkle_model_weights())
  if  model_name == 'effb0':
    wrinkle_model = base_model_efficientNetB0.detection_model('wrinkle')
    wrinkle_model.load_weights(base_model_efficientNetB0.wrinkle_model_weights())

  return wrinkle_model

def acne_model_func(model_name):

  if model_name == "inception":
    acne_model = detection_model('acne')
    acne_model.load_weights(acne_model_weights())
  if model_name == "effb0":
    acne_model = base_model_efficientNetB0.detection_model('acne')
    acne_model.load_weights(base_model_efficientNetB0.acne_model_weights())

  return acne_model

def predictions(model_name):
  """
  This functions returns the coordinates
  of the max possible region that may 
  contain signs of aging (acne, wrinkle)
  
  Input : model_name : inception or effb0  (for InceptionV3 model or efficientNetB0 model) 

  Returns : list_of_rectangles_to_be_drawn 
            example: {
              "File_name_1":(
                            {'example_key1_for_acne': [x1,x2,y1,y2] ,... }, ##Draw {{Blue}} Rectangles
                            {'example_key1_for_wrinkle': [x1,x2,y1,y2] ,... } ##Draw {{Red}} Rectangles
              ), 

               "File_name_2":(
                            {'example_key1_for_acne': [x1,x2,y1,y2] ,... }, ##Draw {{Blue}} Rectangles
                            {'example_key1_for_wrinkle': [x1,x2,y1,y2] ,... } ##Draw {{Red}} Rectangles
                            ) 
            }
        ***Note**: These coordinates are regions
        of image (uploaded image) not the image itself 
  """
  files = glob(IMG_PATH+'/*')
  list_of_rectangles_to_be_drawn = {}

  figure(figsize = (20,16))
  cnt = 1
  for f1 in files:
    img = cvtColor(imread(f1),COLOR_BGR2RGB)
    img_acne = img.copy()
    img_wrinkle = img.copy()
    dimension_dict = {}
    dims, face, img_dim = getPatches.extract_patches(f1,{},{},img.shape,CROPPED_PATCH_DIR)
    print(dims)
    list_potentials_acne = {}
    list_potentials_wrinkle = {}
    for x in dims:

      """Example: dims == {'landmark_fh': [116, 193, 0, 19], 'landmark_chin': [144, 169, 120, 151], 'landmark_rc': [82, 144, 45, 140]}"""
      x1,x2,y1,y2 = dims[x]
      roi = img[y1:y2,x1:x2]
      resized_roi = cv2.resize(roi,(IMG_DIMS[0],IMG_DIMS[1]))
      resized_roi = np.asarray(resized_roi)/255.

      """Dont change any of the rest"""
      roi_tensor = np.expand_dims(resized_roi,0)
      pred_wrinkle = wrinkle_model_func(model_name).predict(roi_tensor)
      pred_acne = acne_model_func(model_name).predict(roi_tensor)
      print(pred_wrinkle)
      print(pred_acne)
      # Confidence comparision can be increased (0.7 --> 0.75 or 0.8 (not preferrable))
      if np.max(pred_wrinkle) >0.7:
        #rectangle(img_wrinkle,(x1,y1),(x2,y2),(255,0,0),2)
        list_potentials_wrinkle[x] = dims[x]
      if np.max(pred_acne) >0.4:
        #rectangle(img_acne, (x1,y1),(x2,y2),(0,0,255),2)
        list_potentials_acne[x] = dims[x]
    list_of_rectangles_to_be_drawn[f1] = (list_potentials_acne,list_potentials_wrinkle)
    """subplot(4,3,cnt)
    cnt+=1
    imshow(img_wrinkle)
    subplot(4,3,cnt)
    cnt+=1
    imshow(img_acne)"""

  return list_of_rectangles_to_be_drawn,"Red,Blue"
