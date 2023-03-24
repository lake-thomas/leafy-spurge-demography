#!/usr/bin/python

import os
import pandas as pd
import numpy as np
import datetime
import pprint
import time
import math
import random
import glob
from functools import reduce
from pprint import pprint


#Plotting
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Tensorflow setup.

# Tensorflow version 2.9.1
import tensorflow as tf
print(tf.__version__)

# Keras setup.
import keras
from keras import layers
from keras.layers import Flatten
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras.regularizers import l2
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, SpatialDropout1D, Concatenate
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint, History, EarlyStopping
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras import backend as K


import sys

# Import from ~/sits folder
# Contains readingsits.py file to read and compute spectral features on SITS
sys.path.append("/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/sits")
import readingsits

# Import from ~/deeplearning folder
# Contains multiple .py files with varying DL architectures 
sys.path.append("/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/deeplearning")

import architecture_features
import architecture_complexity
import architecture_rnn
import architecture_regul
import architecture_batchsize
import architecture_depth
import architecture_spectro_temporal
import architecture_pooling

# Import from ~/outputfiles folder
# Contains evaluation.py and save.py files with fucntions to compute summary statistics, write predictions, and create confusion matrices
sys.path.append("/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/outputfiles")

import evaluation
import save



# Set a model results path
res_path = '/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN'

# Creating output path if does not exist
if not os.path.exists(res_path):
  print("ResPath DNE")
  os.makedirs(res_path)

# Set the path to exported training/testing dataset
sits_path = '/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/datasets_oct22'    
    
# Set Architecture / Model Run Index (used if running in batch on MSI)
noarchi = 0
norun = 0
feature = "SB" #use only spectral bands provided (do not compute new bands, like NDVI, which are already computed)

# Parameters to set
n_channels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2
val_rate = 0.1 # Validation data rate

# Evaluated metrics
eval_label = ['OA', 'train_loss', 'train_time', 'test_time']	
	
# String variables for the training and testing datasets
train_str = 'train_dataset_allyears_full_oct22'
test_str = 'test_dataset_allyears_full_oct22'					

# Get filenames
train_file = sits_path + '/' + train_str + '.csv'
test_file = sits_path + '/' + test_str + '.csv'
print("train_file: ", train_file)
print("test_file: ", test_file)
	
# Output files			
res_path = res_path + '/Archi' + str(noarchi) + '/'
if not os.path.exists(res_path):
  os.makedirs(res_path)
  print("noarchi: ", noarchi)

# Create output files to capture model results
str_result = feature + '-' + train_str + '-noarchi' + str(noarchi) + '-norun' + str(norun) 
res_file = res_path + '/resultOA-' + str_result + '.csv'
res_mat = np.zeros((len(eval_label),1))
traintest_loss_file = res_path + '/trainingHistory-' + str_result + '.csv'
conf_file = res_path + '/confMatrix-' + str_result + '.csv'
out_model_file = res_path + '/bestmodel-' + str_result + '.h5'



from tensorflow.keras.utils import to_categorical

# Read in SITS training and testing datasets
X_train, polygon_ids_train, y_train = readingsits.readSITSData(train_file)
X_test,  polygon_ids_test, y_test = readingsits.readSITSData(test_file)
print(X_test)  #verify spectral band data looks correct
print(X_test.shape) #num_samples, 63 bands (9 timesteps * 7 bands/timestep = 63)


# Number of unique classes in y_train and y_test datasets should = 9
n_classes_test = len(np.unique(y_test))
print(n_classes_test)
n_classes_train = len(np.unique(y_train))
print(n_classes_train)

# heck equal number of classes in training and testing dataset
if(n_classes_test != n_classes_train):
  print("WARNING: different number of classes in train and test")

n_classes = max(n_classes_train, n_classes_test) # 9 classes
y_train_one_hot = to_categorical(y_train) # specify number of classes explicity - may need to recode classes sequentially (1-9) to work correctly?
y_test_one_hot = to_categorical(y_test)

print(y_test_one_hot) #verify one hot encoding was successful
print(y_test_one_hot.shape)
print(y_test_one_hot[0])


X_train = readingsits.addingfeat_reshape_data(X_train, feature, n_channels) #Feature = "SB" (spectral bands)

X_test = readingsits.addingfeat_reshape_data(X_test, feature, n_channels)		

print(X_train[0, :, :])
print(X_train.shape) #verify reshape was successful, now num_samples, num_timesteps, num_bands


#---- Extracting a validation set (if necesary)
if val_rate > 0:
  #Number of samples to take from Training dataset based on validation rate
  val_num_samples = int(math.ceil(X_train.shape[0] * val_rate))

  #Select random indices for val_num_samples to select validation set
  val_indices = random.sample(range(1, X_train.shape[0]), val_num_samples)
  #remove these indices from the training set
  train_indices = np.delete(range(1, X_train.shape[0]), val_indices)

  #Create training and validation sets 
  X_val = X_train[val_indices, :]
  y_val = y_train[val_indices]
  X_train = X_train[train_indices, :]
  y_train = y_train[train_indices]

  #--- Computing the one-hot encoding (recomputing it for train)
  y_train_one_hot = to_categorical(y_train)
  y_val_one_hot = to_categorical(y_val)

  n_classes_val = len(np.unique(y_val))
  print(n_classes_val)
  n_classes_train = len(np.unique(y_train))
  print(n_classes_train)

  #Check equal number of classes in training and testing dataset
  if(n_classes_val != n_classes_train):
    print("WARNING: different number of classes in train and test")
  

print(X_train.shape, y_train_one_hot.shape, X_val.shape, y_val_one_hot.shape, X_test.shape, y_test_one_hot.shape)


#-----------------------------------------------------------------------		
def conv_bn(X, **conv_params):	
	nbunits = conv_params["nbunits"];
	kernel_size = conv_params["kernel_size"];

	strides = conv_params.setdefault("strides", 1)
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

	Z = Conv1D(nbunits, kernel_size=kernel_size, 
			strides = strides, padding=padding,
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer)(X)

	return BatchNormalization(axis=-1)(Z) #-- CHANNEL_AXIS (-1)

#-----------------------------------------------------------------------		
def conv_bn_relu(X, **conv_params):
	Znorm = conv_bn(X, **conv_params)
	return Activation('relu')(Znorm)
	
#-----------------------------------------------------------------------		
def conv_bn_relu_drop(X, **conv_params):	
	dropout_rate = conv_params.setdefault("dropout_rate", 0.5)
	A = conv_bn_relu(X, **conv_params)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def fc_bn(X, **fc_params):
	nbunits = fc_params["nbunits"];
	
	kernel_regularizer = fc_params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = fc_params.setdefault("kernel_initializer", "he_normal")
		
	Z = Dense(nbunits, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(X)
	return BatchNormalization(axis=-1)(Z) #-- CHANNEL_AXIS (-1)
	
#-----------------------------------------------------------------------		
def fc_bn_relu(X, **fc_params):	
	Znorm = fc_bn(X, **fc_params)
	return Activation('relu')(Znorm)

#-----------------------------------------------------------------------		
def fc_bn_relu_drop(X, **fc_params):
	dropout_rate = fc_params.setdefault("dropout_rate", 0.5)
	A = fc_bn_relu(X, **fc_params)
	return Dropout(dropout_rate)(A)

#-----------------------------------------------------------------------		
def softmax(X, nbclasses, **params):
	kernel_regularizer = params.setdefault("kernel_regularizer", l2(1.e-6))
	kernel_initializer = params.setdefault("kernel_initializer", "glorot_uniform")
	return Dense(nbclasses, activation='softmax', 
			kernel_initializer=kernel_initializer,
			kernel_regularizer=kernel_regularizer)(X)



#Get input sizes
m, L, depth = X_train.shape
input_shape = (L, depth)
#or input_shape = (63, 1) ?

#-- parameters of the architecture
nbclasses = 10


l2_rate = 1.e-6
dropout_rate = 0.1
nb_conv = 3
nb_fc= 1
nbunits_conv = 64 #-- will be double
nbunits_fc = 256 #-- will be double

	# Define the input placeholder.
X_input = Input(input_shape)
		
	#-- nb_conv CONV layers
X = X_input
for add in range(nb_conv):
    X = conv_bn_relu_drop(X, nbunits=nbunits_conv, kernel_size=5, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
	#-- Flatten + 	1 FC layers
X = Flatten()(X)
for add in range(nb_fc):	
    X = fc_bn_relu_drop(X, nbunits=nbunits_fc, kernel_regularizer=l2(l2_rate), dropout_rate=dropout_rate)
		
	#-- SOFTMAX layer
out = softmax(X, nbclasses, kernel_regularizer=l2(l2_rate))
		
	# Create model.

model = Model(inputs = X_input, outputs = out, name='Archi_3CONV64_1FC256')	





#Define Class Weights
from sklearn.utils import class_weight

class_weights =  class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train                                                   
                                    )
class_weights = dict(zip(np.unique(y_train), class_weights))

# Class weights function, 
class_weights = {0: 0,
                 1: 7.046028630719989,
                 2: 3.6421837069230087,
                 3: 31.37461158722999,
                 4: 0.7614511317372198,
                 5: 0.6015453322153169,
                 6: 0.3652990948014909,
                 7: 0.39487324200412083,
                 8: 4.334510403657227,
                 9: 20.275284755853498}

print(class_weights)


###
# Define Model Variables
###

# Model variables
n_epochs = 1000
batch_size = 512
lr = 0.0001 #recommended in Allred et al., 2021
beta_1 = 0.9 #not used, but can be used to modify optimizer LR
beta_2 = 0.98
decay = 0.0
	
#Model Optimizer
#opt = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
opt = tf.keras.optimizers.Adam(lr=lr)

# Compile Model
model.compile(optimizer = opt, loss = "mean_squared_error", metrics = ["accuracy"])

model.summary()




# Model callbacks
checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')


start_train_time = time.time()

# Fit the model
hist = model.fit(x = X_train, 
                 y = y_train_one_hot, 
                 epochs = n_epochs,
                 batch_size = batch_size, 
                 shuffle=True,
                 validation_data=(X_val, y_val_one_hot), 
                 verbose=2,
                 callbacks = [])


train_time = round(time.time()-start_train_time, 2)

# Save the Trained Model as a .h5 file
model.save(r'/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/Archi0/draft_cnn_modeL_300epochs_oct272022.h5')

from sklearn.metrics import multilabel_confusion_matrix
from tabulate import tabulate

# Predict the model on withheld testing dataset
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=-1)
y_pred_flat = y_pred.flatten()
y_pred_flat = y_pred_flat.astype(int)

y_test = y_test.astype(int)    
y_test_flat = y_test.flatten()


# Calculate confusion matrix
class_names = ["Water", "Developed", "BarrenLand", "Forest", "Shrub/Scrub", "Grassland/Herbaceous", "Croplands", "EmergentWetlands", "LeafySpurge"]
class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
c = multilabel_confusion_matrix(y_test_flat, y_pred_flat, labels = class_labels)
model_output_metrics = []
for i in range(len(class_labels)):
    tn=c[i, 0, 0]
    tp=c[i, 1, 1]
    fn=c[i, 1, 0]
    fp=c[i, 0, 1]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    TPR_Sens_Recall = tp/(tp+fn)
    TNR_Spec = tn/(tn+fp)
    FPR = fp/(fp+tn)
    FNR = fn/(fn+tp)
    precision = tp/(tp+fp)
    jaccard = tp/(tp+fp+fn)
    beta = 0.5
    F05 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    beta = 1
    F1 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    beta = 2
    F2 = ((1 + beta**2) * precision * TPR_Sens_Recall) / (beta**2 * precision + TPR_Sens_Recall)
    outputs = [class_names[i], tp, tn, fp, fn, accuracy, TPR_Sens_Recall, TNR_Spec, FPR, FNR, precision, jaccard, F1]
    model_output_metrics.append(outputs)

# Print and format outputs
print(tabulate(model_output_metrics, floatfmt=".2f", headers=["Class Name", "TP", "TN", "FP", "FN", "Accuracy", "TPR/Sens/Recall", "TNR/Spec", "FPR", "FNR", "Precision", "Jaccard", "F1"]))

  
#EOF