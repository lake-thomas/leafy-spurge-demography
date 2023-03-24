#!/usr/bin/python

#Script uses conda environment 'tf_gpu_earthengine' from conda env 'tf_gpu'


# Specify Arguments for file input (to select bounding box and extract training datasets)
from sys import argv

# Argument (interger) specifies class weight for leafy spurge class (experimental)
spurge_weight = int(argv[1])

# Packages
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


# Plotting
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Tensorflow setup

# Tensorflow version 2.4.1
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
# Contains readingsits.py file to read and compute and reshape the SITS data
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

# Set the path to exported training/testing dataset
sits_path = '/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/datasets_oct22'    
    
    
#-----------------------------------------------------------------------
# Parameters to set
n_channels = 7 #-- B G NDVI NIR Red SWIR1 SWIR2
val_rate = 0.1 # Validation data rate

# String variables for the training and testing datasets
train_str = 'train_dataset_allyears_full_oct22'
test_str = 'test_dataset_allyears_full_oct22'					

# Get filenames
train_file = sits_path + '/' + train_str + '.csv'
test_file = sits_path + '/' + test_str + '.csv'
print("train_file: ", train_file)
print("test_file: ", test_file)

# Set a model results path
res_path = '/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN'

# Set Architecture / Model Run Index (used if running in batch on MSI)
noarchi = 2
norun = 1
feature = "SB" #use only spectral bands provided (do not compute new bands, like NDVI, which are already computed)

# Creating output path if does not exist
if not os.path.exists(res_path):
  print("ResPath DNE")
  os.makedirs(res_path)

# Output files			
res_path = res_path + '/Archi' + str(noarchi) + '/'
if not os.path.exists(res_path):
  os.makedirs(res_path)
  print("noarchi: ", noarchi)

    
# Create output files to capture model results
str_result = feature + '-' + train_str + '-noarchi' + str(noarchi) + '-norun-' + str(norun) + '-spurge_weight-' + str(spurge_weight)

# Output for model evaluation metrics
res_file = res_path + 'result_accuracy_metrics-' + str_result + '.txt'

# Output for model loss / epochs
traintest_loss_file = res_path + 'trainingHistory-' + str_result + '.txt'

# Output for confusion matrix
conf_file = res_path + 'confMatrix-' + str_result + '.txt'

# Output for model weights file (.h5 file)
out_model_file = res_path + 'model-' + str_result + '.h5'


print(res_file)
print(traintest_loss_file)
print(conf_file)
print(out_model_file)



#-----------------------------------------------------------------------
from tensorflow.keras.utils import to_categorical

# Read in SITS training and testing datasets
X_train, polygon_ids_train, y_train = readingsits.readSITSData(train_file)
X_test,  polygon_ids_test, y_test = readingsits.readSITSData(test_file)
#print(X_test)  #verify spectral band data looks correct
print(X_train.shape) #num_samples, 63 bands (9 timesteps * 7 bands/timestep = 63)

# Number of unique classes in y_train and y_test datasets should = 9
n_classes_test = len(np.unique(y_test))
print(n_classes_test)
n_classes_train = len(np.unique(y_train))
print(n_classes_train)

# Check equal number of classes in training and testing dataset
if(n_classes_test != n_classes_train):
  print("WARNING: different number of classes in train and test")

n_classes = max(n_classes_train, n_classes_test) # 9 classes + 1 background class (when one-hot encoding)
y_train_one_hot = to_categorical(y_train) # specify number of classes explicity - may need to recode classes sequentially (1-9) to work correctly?
y_test_one_hot = to_categorical(y_test)

#print(y_test_one_hot) #verify one hot encoding was successful
print(y_train_one_hot.shape)
#print(y_test_one_hot[0])

X_train = readingsits.addingfeat_reshape_data(X_train, feature, n_channels) #Feature = "SB" (spectral bands)

X_test = readingsits.addingfeat_reshape_data(X_test, feature, n_channels)		

print(X_train[0, :, :])
print(X_train.shape) #verify reshape was successful, now num_samples, num_timesteps, num_bands


#-----------------------------------------------------------------------
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

#-- parameters of the architecture
nbclasses = 10
l2_rate = 1.e-6
dropout_rate = 0.1
nb_conv = 3
nb_fc= 1
nbunits_conv = 64 #-- will be double
nbunits_fc = 128 #-- will be double

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



###
# Define Model Variables
###

# Model variables
n_epochs = 100
batch_size = 10192
lr = 0.0001 #recommended in Allred et al., 2021
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-07

#Define Class Weights
from sklearn.utils import class_weight

class_weights =  class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)

class_weights = dict(zip(np.unique(y_train), class_weights))

# Class weights function
# inverse of frequency
class_weights = {0: 0,
                 1: 1.046028630719989,
                 2: 1.6421837069230087,
                 3: 1.37461158722999,
                 4: 0.7614511317372198,
                 5: 0.6015453322153169,
                 6: 0.3652990948014909,
                 7: 0.39487324200412083,
                 8: 4.334510403657227,
                 9: spurge_weight}

print(class_weights)


# squared of inverse frequenecy weights?


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

            
            
# Number of training samples.
sample_count = X_train.shape[0]

# Total epochs to train.
epochs = n_epochs

# Number of warmup epochs. (10% of total epochs)
warmup_epoch = 10

# Training batch size, set small value here for demonstration purpose.
batch_size = batch_size

# Base learning rate after warmup.
learning_rate_base = 0.001

total_steps = int(epochs * sample_count / batch_size) #98,275

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size) #9827

# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)






#Model Optimizer
opt = tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Compile Model
model.compile(optimizer = opt, loss = cce, metrics = ["accuracy"])

model.summary()

#Train the model
start_train_time = time.time()

# Fit the model
hist = model.fit(x = X_train, 
                 y = y_train_one_hot, 
                 epochs = n_epochs,
                 batch_size = batch_size, 
                 shuffle=True,
                 validation_data=(X_val, y_val_one_hot), 
                 verbose=1,
                 class_weight = class_weights,
                callbacks=[warm_up_lr])


train_time = round(time.time()-start_train_time, 2)



# Save the Trained Model as a .h5 file
model.save(out_model_file)
           

# Evaluate the model prediction

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

#Save model results to file
with open(res_file, 'w') as f:
    f.write(tabulate(model_output_metrics, floatfmt=".2f", headers=["Class Name", "TP", "TN", "FP", "FN", "Accuracy", "TPR/Sens/Recall", "TNR/Spec", "FPR", "FNR", "Precision", "Jaccard", "F1"]))
    

# Save losses and accuracy
train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['accuracy']
val_acc    = hist.history['val_accuracy']
xc         = range(n_epochs)

traintest_loss_df = pd.DataFrame(
    {'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'epochs': xc\
    })

traintest_loss_df
    
np.savetxt(traintest_loss_file, traintest_loss_df, fmt='%6f')



import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread, imshow, imsave
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, fbeta_score
from tensorflow.keras.models import load_model
from tabulate import tabulate

def plot_confusion_matrix(
        y_true,
        y_pred,
        classes,
        test_name,
        normalize=False,
        set_title=False,
        save_fig=False,
        cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if set_title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # and save it to log file
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        #with open(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_log.txt', 'ab') as f:
        #    f.write(b'\nNormalized confusion matrix\n')
        #    np.savetxt(f, cm, fmt='%.3f')
    else:
        print('Confusion matrix, without normalization')
        #with open(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_log.txt', 'ab') as f:
        #    f.write(b'\nConfusion matrix, without normalization\n')
        #    np.savetxt(f, cm, fmt='%7u')

    #print(cm)
    #cm = cm[1:10]
    #cm = cm[:,1:]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if normalize:
        im.set_clim(0., 1.)     # fixes missing '1.0' tick at top of colorbar
    cb = ax.figure.colorbar(im, ax=ax)
    if normalize:
        cb.set_ticks(np.arange(0., 1.2, 0.2))
        cb.set_ticklabels([f'{i/5:.1f}' for i in range(6)])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title if set_title else None,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(cm)-0.5, -0.5)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if np.round(cm[i, j], 2) > 0.:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, '–',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_fig:
        if normalize:
            plt.savefig(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_cm_normal.pdf')
        else:
            plt.savefig(f'F:/PlanetScope_LSTM_Imagery/reports/logs_and_plots/{test_name}_cm_non_normal.pdf')
    return fig, ax, cm




inv_category_dict = {1:"Water", 2:"Developed", 3:"BarrenLand", 4:"Forest", 5:"Shrub/Scrub", 6:"Grassland/Herbaceous", 7:"Croplands", 8:"EmergentWetlands", 9:"LeafySpurge"}

class_names = [inv_category_dict[i] for i in np.arange(1, 10)]


# save plot of normalized cm
cm = plot_confusion_matrix(
    y_test_flat,
    y_pred_flat,
    classes=class_names,
    test_name="myModel",
    normalize=True,
    save_fig=False
)

conf_df = pd.DataFrame(cm[2], columns = ["Water", "Developed", "BarrenLand", "Forest", "Shrub/Scrub", "Grassland/Herbaceous", "Croplands", "EmergentWetlands", "LeafySpurge"])

np.savetxt(conf_file, conf_df, fmt='%6f')


#EOF
