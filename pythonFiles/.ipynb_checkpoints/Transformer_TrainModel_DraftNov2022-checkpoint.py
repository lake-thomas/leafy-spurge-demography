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

# Tensorflow version 2.4.1
import tensorflow as tf
print(tf.__version__) 

from tensorflow import keras
from tensorflow.keras import layers

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


#Format of X and Y training data for input in Transformer model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)



###
# Define Model Variables
###

# Model variables
n_epochs = 100
batch_size = 5000

input_shape = X_train.shape[1:]

#Plot Loss and Accuracy Callback
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.f1 = []
        self.val_f1 = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.f1.append(logs.get('accuracy'))
        self.val_f1.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val loss")
        ax1.legend()
        
        ax2.plot(self.x, self.f1, label="Acc")
        ax2.plot(self.x, self.val_f1, label="val Acc ")
        ax2.legend()
        
        plt.show();
        
plot_losses = PlotLearning()

# Learning Rate Warmup and Decay Callback
def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)
        
        
# Define the warmup callback        
# 5% of the steps
warmup_steps = int(0.05*n_epochs)
warmup_callback = WarmupCosineDecay(total_steps=n_epochs, 
                             warmup_steps=warmup_steps,
                             hold=int(warmup_steps/2), 
                             start_lr=0.0, 
                             target_lr=1e-3)

# Define class weights
# inverse of frequency
class_weights = {0: 0,
                 1: 7.046028630719989,
                 2: 3.6421837069230087,
                 3: 31.37461158722999,
                 4: 0.7614511317372198,
                 5: 0.6015453322153169,
                 6: 0.3652990948014909,
                 7: 0.39487324200412083,
                 8: 4.334510403657227,
                 9: 13.275284755853498}


model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(
    loss = "categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

model.summary()

# Model callbacks
#checkpoint = ModelCheckpoint(out_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', restore_best_weights=True)
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')

start_train_time = time.time()

hist = model.fit(x = X_train,
                 y = y_train_one_hot,
                 epochs = n_epochs,
                 batch_size = batch_size,
                 shuffle=True,
                 validation_data=(X_val, y_val_one_hot),
                 verbose=1,
                 callbacks=[plot_losses, warmup_callback],
                 class_weight=class_weights)

train_time = round(time.time()-start_train_time, 2)

print(train_time)


# Save the Trained Model as a .h5 file
model.save(r'/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/Archi0/draft_transformer_modeL_100epochs_lrsch_5kbatch_nov172022.h5')

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