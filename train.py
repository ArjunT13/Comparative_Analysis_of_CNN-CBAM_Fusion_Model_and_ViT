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
from cnn_cbam import buildModel


gpu_fraction = 1
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def preprocess_vit(data, outcome, ehr, image_name, file_name):
    img_arr, label, ehr_arr = [], [], []
    with open(file_name, 'r') as file:
        for line in file:
            i=line.strip()
            if i[:-4] in image_name:
                img_file =Path('/mnt/storage/EGC_outcome/EKG/POST/'+i)
                img = image.load_img('/mnt/storage/EGC_outcome/EKG/POST/'+i, color_mode='rgb')
                # print(img.size)
                j = data.index[data['POST'] == i[:-4]].tolist()
                height, width = img.size
                trim1, trim2=10, 20
                if(height<=20 or width<=20):
                    print(data['POST'].iloc[i])
                    trim1 = 1
                    trim2 = 2
                col1_x=img.size[0]/4+trim1
                col2_x=col1_x+img.size[0]/4-trim2
                col3_x=col2_x+img.size[0]/4-trim2
                col4_x=img.size[0]

                row_y=img.size[1]/2
                row1_y=row_y/3 + trim1
                row2_y=row1_y + row_y/3
                row3_y=row2_y + row_y/3
                row4_y=row3_y + row_y/3
                row5_y=row4_y + row_y/3
                row6_y=row5_y + row_y/3
                """Dictionary to choose the specific lead(s) to crop from the ECG image"""
                crop_dict={ 
                'Lead1':(trim1,0,col1_x,row1_y), 'Lead2':(col1_x,0,col2_x,row1_y), 'Lead3':(col2_x,0,col3_x,row1_y), 'Lead4':(col3_x,0,col4_x,row1_y),
                'Lead5':(trim1,row1_y,col1_x,row2_y), 'Lead6':(col1_x,row1_y,col2_x,row2_y),'Lead7':(col2_x,row1_y,col3_x,row2_y), 'Lead8':(col3_x,row1_y,col4_x,row2_y),
                'Lead9':(trim1,row2_y,col1_x,row3_y), 'Lead10':(col1_x,row2_y,col2_x,row3_y), 'Lead11':(col2_x,row2_y,col3_x,row3_y), 'Lead12':(col3_x,row2_y,col4_x,row3_y),
                'RhythmLead1':(trim1,row3_y,col4_x,row4_y), 'RhythmLead2':(trim1,row4_y,col4_x,row5_y), 'RhythmLead3':(trim1,row5_y,col4_x,row6_y), 'Lead_combined':(trim1,row4_y,col4_x,row6_y)
                }
                for row_idx in j:
                    label.append(data[outcome][row_idx])
                    ehr_arr.append(ehr[image_name.index(i[:-4])])
                    img_res = img.crop(crop_dict['Leadcombined'])
                    bnwimg = np.array(img_res)
                    bnwimg.resize(124, 124, 3)
                    img_arr.append(bnwimg)
    return np.array(img_arr), np.array(label), np.array(ehr_arr)


def preprocess_cnn(data, outcome, ehr, image_name, file_name):
    img_arr, label, ehr_arr = [], [], []
    with open(file_name, 'r') as file:
        for line in file:
            i=line.strip()
            if i[:-4] in image_name:
                img_file =Path('/mnt/storage/EGC_outcome/EKG/POST/'+i)
                img = image.load_img('/mnt/storage/EGC_outcome/EKG/POST/'+i, color_mode='rgb')
                j = data.index[data['POST'] == i[:-4]].tolist()
                # print(j, end=" ---- ")
                for row_idx in j:
                    label.append(data[outcome][row_idx])
                    ehr_arr.append(ehr[image_name.index(i[:-4])])
                    img_r = np.array(img)
                    # (thresh, bnwimg) = cv2.threshold(img_r, 160, 255, cv2.THRESH_BINARY)
                    # bnwimg.resize(125, 125, 1)
                    img_arr.append(img_r)
                    # print("Image "+i+" loaded..")
    img_arr, label, ehr_arr = np.array(img_arr), np.array(label), np.array(ehr_arr)
    img_lead1, img_lead2 = [], []
    ct=0
    for ecg_img in img_arr:
        ecg_img = Image.fromarray(ecg_img)
        height, width = ecg_img.size
        trim1, trim2=10, 20
        if(height<=20 or width<=20):
            ct+=1
            trim1 = 1
            trim2 = 2
        col1_x=ecg_img.size[0]/4+trim1
        col2_x=col1_x+ecg_img.size[0]/4-trim2
        col3_x=col2_x+ecg_img.size[0]/4-trim2
        col4_x=ecg_img.size[0]

        row_y=ecg_img.size[1]/2
        row1_y=row_y/3 + trim1
        row2_y=row1_y + row_y/3
        row3_y=row2_y + row_y/3
        row4_y=row3_y + row_y/3
        row5_y=row4_y + row_y/3
        row6_y=row5_y + row_y/3

        """Dictionary to choose the specific lead(s) to crop from the ECG image"""
        crop_dict={ 
        'Sign1':(trim1,0,col1_x,row1_y), 'Sign2':(col1_x,0,col2_x,row1_y), 'Sign3':(col2_x,0,col3_x,row1_y), 'Sign4':(col3_x,0,col4_x,row1_y),
        'Sign5':(trim1,row1_y,col1_x,row2_y), 'Sign6':(col1_x,row1_y,col2_x,row2_y),'Sign7':(col2_x,row1_y,col3_x,row2_y), 'Sign8':(col3_x,row1_y,col4_x,row2_y),
        'Sign9':(trim1,row2_y,col1_x,row3_y), 'Sign10':(col1_x,row2_y,col2_x,row3_y), 'Sign11':(col2_x,row2_y,col3_x,row3_y), 'Sign12':(col3_x,row2_y,col4_x,row3_y),
        'Sign13':(trim1,row3_y,col4_x,row4_y), 'Sign14':(trim1,row4_y,col4_x,row5_y), 'Sign15':(trim1,row5_y,col4_x,row6_y)
        }

        """Currently considering the 10 second long rhythm strips V1 and V5"""
        for j in range(1, 16):
            img_res1 = ecg_img.crop(crop_dict['Sign'+str(j)])
            bnwimg1 = np.array(img_res1)
            bnwimg1.resize(124, 124, 3)
            if(j==14):
                img_lead1.append(bnwimg1)
            elif(j==15):
                img_lead2.append(bnwimg1)
    
    return np.array(img_lead1), np.array(img_lead2), label, ehr_arr



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
    
    # outcomes =  [ 'Time from PCI to Stroke_1yr', 'Time from PCI to CHF Hospitalization_1yr', Mortality 1 yr\nYes:1\nNo:0']
    for outcome in outcomes:
        
        """ If using the 1 yr data labels remove the comments from the lines below """
        
        # if outcome == 'Time from PCI to Stroke_6mo':
        #     outcome = 'Time from PCI to Stroke_1yr'
        # elif outcome == 'Time from PCI to CHF Hospitalization_6mo':
        #     outcome = 'Time from PCI to CHF Hospitalization_1yr'
        # elif outcome == 'Mortality 6 months\nYes:1\nNo:0':
        #     outcome = 'Mortality 1 yr\nYes:1\nNo:0'

        print("Generating results for: ", outcome)
        
        demo, image_name = load_EHR_data(data, outcome)

        if outcome == 'Time from PCI to Stroke_6mo' or outcome == 'Time from PCI to Stroke_1yr' :
            train_file_name = 'image_name_train Time from PCI to Stroke_1yr.txt'
            test_file_name = 'image_name_val Time from PCI to Stroke_1yr.txt'
        elif outcome == 'Time from PCI to CHF Hospitalization_6mo' or  outcome == 'Time from PCI to CHF Hospitalization_1yr':
            train_file_name = 'image_name_train Time from PCI to CHF Hospitalization_1yr.txt'
            test_file_name = 'image_name_val Time from PCI to CHF Hospitalization_1yr.txt'
        elif outcome == 'Mortality 6 months\nYes:1\nNo:0' or outcome == 'Mortality 1 yr\nYes:1\nNo:0':
            train_file_name = 'image_name_train_mortality.txt'
            test_file_name = 'image_name_val_mortality.txt'
        
        """ECG Data Preprocessing"""
        trainX, trainY, tab_train = preprocess_vit(data, outcome, demo, image_name, train_file_name)
        testX, testY, tab_test = preprocess_vit(data, outcome, demo, image_name, test_file_name)
        print("Train Set: ", trainX.shape, trainY.shape)
        print("Test Set: ", testX.shape, testY.shape)
        print("EHR Data: - Train: ", tab_train.shape, " Test: ", tab_test.shape)

        """Define the Model"""
        model = vit_model(trainX) 

        """Taking class_weights since its imbalanced dataset"""
        class_weights = compute_class_weight('balanced', classes=np.unique(trainY), y=trainY)
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

        """Defining the callbacks"""
        csv_log_path = outcome+"_training_logs_vit.csv"
        csv_logger = CSVLogger(csv_log_path, append=True)
        model1_es = EarlyStopping(monitor = 'loss', min_delta = 1e-11, patience = 10, verbose = 1)
        model1_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 6, verbose = 1)

        # Automatically saves the best weights of the model, based on best val_accuracy
        model1_mcp = ModelCheckpoint(filepath = outcome+'_vit_model_weights.h5', monitor = 'loss', save_best_only = True, verbose = 1)
        
        """Training the model"""
        history = model.fit([tab_train, trainX], trainY, epochs=50, batch_size=16, validation_split=0.2, class_weight=class_weights_dict, callbacks=[model1_es, model1_rlr, csv_logger, model1_mcp])
        
        predY = model.predict([tab_test, testX])


        """Storing the labels"""
        true_variables.append(trainY)
        pred_variables.append(predY)
        
        
        """Storing the model"""        
        models.append(model)
        
    return true_variables, pred_variables, outcomes, models


def train_CNN_CBAM_Model(data):
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
    
    # outcomes =  [ 'Time from PCI to Stroke_1yr', 'Time from PCI to CHF Hospitalization_1yr', Mortality 1 yr\nYes:1\nNo:0']
    for outcome in outcomes:
        
        """ If using the 1 yr data labels remove the comments from the lines below """
        
        # if outcome == 'Time from PCI to Stroke_6mo':
        #     outcome = 'Time from PCI to Stroke_1yr'
        # elif outcome == 'Time from PCI to CHF Hospitalization_6mo':
        #     outcome = 'Time from PCI to CHF Hospitalization_1yr'
        # elif outcome == 'Mortality 6 months\nYes:1\nNo:0':
        #     outcome = 'Mortality 1 yr\nYes:1\nNo:0'

        print("Generating results for: ", outcome)
        
        demo, image_name = load_EHR_data(data, outcome)

        if outcome == 'Time from PCI to Stroke_6mo' or outcome == 'Time from PCI to Stroke_1yr' :
            train_file_name = 'image_name_train Time from PCI to Stroke_1yr.txt'
            test_file_name = 'image_name_val Time from PCI to Stroke_1yr.txt'
        elif outcome == 'Time from PCI to CHF Hospitalization_6mo' or  outcome == 'Time from PCI to CHF Hospitalization_1yr':
            train_file_name = 'image_name_train Time from PCI to CHF Hospitalization_1yr.txt'
            test_file_name = 'image_name_val Time from PCI to CHF Hospitalization_1yr.txt'
        elif outcome == 'Mortality 6 months\nYes:1\nNo:0' or outcome == 'Mortality 1 yr\nYes:1\nNo:0':
            train_file_name = 'image_name_train_mortality.txt'
            test_file_name = 'image_name_val_mortality.txt'
        
        """ECG Data Preprocessing"""
        trainX_lead1, trainX_lead2, trainY, tab_train = preprocess_cnn(data, outcome, demo, image_name, train_file_name)
        testX_lead1, testX_lead2, testY, tab_test = preprocess_cnn(data, outcome, demo, image_name, test_file_name)
        print("Train Set: ", trainX_lead1.shape, trainX_lead2.shape, trainY.shape)
        print("Test Set: ", testX_lead1.shape, testX_lead2.shape, testY.shape)
        print("EHR Data: - Train: ", tab_train.shape, " Test: ", tab_test.shape)

        """Define the Model"""
        model = buildModel(157) 

        """Taking class_weights since its imbalanced dataset"""
        class_weights = compute_class_weight('balanced', classes=np.unique(trainY), y=trainY)
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

        """Defining the callbacks"""
        csv_log_path = outcome+"_training_logs_cnn_cbam.csv"
        csv_logger = CSVLogger(csv_log_path, append=True)
        model1_es = EarlyStopping(monitor = 'loss', min_delta = 1e-11, patience = 10, verbose = 1)
        model1_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 6, verbose = 1)

        # Automatically saves the best weights of the model, based on best val_accuracy
        model1_mcp = ModelCheckpoint(filepath = outcome+'_cnn_cbam_model_weights.h5', monitor = 'loss', save_best_only = True, verbose = 1)
        
        """Training the model"""
        history = model.fit([tab_train, trainX_lead1, trainX_lead2], trainY, epochs=50, batch_size=16, validation_split=0.2, class_weight=class_weights_dict, callbacks=[model1_es, model1_rlr, csv_logger, model1_mcp])
        
        predY = model.predict([tab_test, testX_lead1, testX_lead2])


        """Storing the labels"""
        true_variables.append(trainY)
        pred_variables.append(predY)
        
        
        """Storing the model"""        
        models.append(model)
        
    return true_variables, pred_variables, outcomes, models