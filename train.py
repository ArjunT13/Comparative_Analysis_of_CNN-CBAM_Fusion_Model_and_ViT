import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path, sys, re
from PIL import Image
import keras.utils as image
from pathlib import Path
import tensorflow as tf
import keras
# import tensorflow_addons as tfa
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Concatenate, BatchNormalization, Dropout, Attention
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import layers

from inputread import load_EHR_data
from vit import vit_model


gpu_fraction = 1
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def train_ViT_Model(data):
    # Storing the outcomes 
    outcomes = ['Time from PCI to Stroke_6mo','Time from PCI to CHF Hospitalization_6mo','Mortality 6 months\nYes:1\nNo:0']
    true_variables= []
    pred_variables =[]
    models=[]

    """
       true_variables: an array containing the labels 
       pred_variables: an array containing the predicted results for the outcomes 
       models: A list of the models trained. """           

    """If using the 1 yr data as label, remove the comment for the code below"""
    
    # outcomes2 =  [ 'Time from PCI to Stroke_1yr', 'Time from PCI to CHF Hospitalization_1yr', Mortality 1 yr\nYes:1\nNo:0']
    for outcome in outcomes:
        
        print("Generating results for: ", outcome)
        """ If using the 1 yr data labels remove the comments from the lines below """
        
        # if outcome == 'Time from PCI to Stroke_6mo':
        #     outcome2 = 'Time from PCI to Stroke_1yr'
        # elif outcome == 'Time from PCI to CHF Hospitalization_6mo':
        #     outcome2 = 'Time from PCI to CHF Hospitalization_1yr'
        # elif outcome == 'Mortality 6 months\nYes:1\nNo:0':
        #     outcome2 = 'Mortality 1 yr\nYes:1\nNo:0'
        
        
        demo, image_name = load_EHR_data(data, outcome)


        



        
        
    #     model = vit_model(trainX)   
        
        
    #     """Taking the class_weights since its imbalanced dataset"""
        
        
    #     class_weights = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(labels_train),
    #                                                  labels_train)
        
                                                     
                                                     
    #     """Defining the callbacks"""
    #     callback_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    #     callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    #     tf.config.run_functions_eagerly(True)
        
        
    
    #     """Training the model"""

    #     weight = {i : class_weights[i] for i in range(2)}
       

    #     history = model.fit([tab_train,ecg_train], labels_train, batch_size=64, validation_split = 0.2, epochs = 100,
    #                         callbacks=[callback_stopping, callback_scheduler],shuffle=True,class_weight=weight) 
    
    #     #Storing the actual and predicted labels for the outcomes 
        
        
    #     preds = model.predict([tab_test,ecg_test])
        
    #     """Storing the labels"""
        
    #     true_variables.append(labels_test)
    #     pred_variables.append(preds)
        
        
    #     """Storing the model"""
        
    #     models.append(model)
        
    # return true_variables, pred_variables, outcomes, models