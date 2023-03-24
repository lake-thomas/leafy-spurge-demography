#!/usr/bin/python

#Load packages

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

#Prediction
import rasterio
from rasterio.plot import show
import glob
import time

# Specify Arguments for file input (which raster to predict
from sys import argv

# Argument (interger) specifies what raster to predict in a list of rasters
input_value = int(argv[1])

# Load a trained model
model = keras.models.load_model(r'/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/temporalCNN/Archi1/TemporalCNN_100epochs_10192_warmup_smoothing_weighted_nov172022.h5')

# Input prediction .tif path
image_path = r'/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/landsat_tifs_2019/'

# Output prediction file path
outpath = r'/panfs/roc/groups/7/moeller/shared/leafy-spurge-demography/datasets_oct22/raster_predictions_2019'

# List all .tif files in /rasters folder for prediction
tif_image_list = glob.glob(image_path + '*.tif')

print(tif_image_list[input_value])

#record how long a prediction takes
prediction_train_time = time.time()

# Open .tif array image with rasterio, read to numpy array
with rasterio.open(tif_image_list[input_value], 'r') as ds:
    arr = ds.read()  # read all raster values

# Define shape of input .tif image
bands, width, height = arr.shape

# Convert Data Type to float32 by division.
arr = arr/10000

# Reshape .tif array axes for correct format so model can predict.
arr = np.moveaxis(arr, 0, -1) #move axis to channels last
new_arr = arr.reshape(-1, arr.shape[-1]) #reshape to row and column
num_pixels = width*height
new_arr2 = new_arr.reshape(num_pixels, 9, 7)
print(new_arr2.shape)

# Predict model and reshape to export.
p = model.predict(new_arr2) # p is prediction from the DL model
pim = p.reshape(width, height, 10) # Dimension of prediction in rows, columns, bands (10 classes)
pim2 = np.moveaxis(pim, 2, 0) # move axis so bands is first

# ArgMax for Segmentation.
pim3 = np.argmax(pim2, axis=0) # take softmax of predictions for segmentation
print(pim3.shape)

# Get the file name (landsat_image_170_t.tif) by splitting input path.
fileout_string = os.path.split(tif_image_list[input_value])

# Output prediction raster .
out_meta = ds.meta.copy()

# Get Output metadata.
out_meta.update({'driver':'GTiff',
                 'width':ds.shape[1],
                 'height':ds.shape[0],
                 'count':1,
                 'dtype':'float64',
                 'crs':ds.crs, 
                 'transform':ds.transform,
                 'nodata':0})

# Write predicted raster to file.
with rasterio.open(fp=outpath + "/prediction_" + fileout_string[-1], #outputpath_name
             mode='w',**out_meta) as dst:
             dst.write(pim3, 1) # the numer one is the number of bands

print("Writing file..." + fileout_string[-1])

prediction_time = round(time.time()-prediction_train_time, 2)

print(prediction_time)

#EOF