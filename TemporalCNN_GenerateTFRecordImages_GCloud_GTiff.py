#!/usr/bin/python

# Python script to process Landsat satellite image mosaics and save to Google Cloud

# Specify Arguments for file input (to select bounding box and extract training datasets)
from sys import argv

# Argument (interger) specifies what bounding box to select for sampling data
input_value = int(argv[1])

# Connect GCE service account to Earth engine API
# Note: Accessing EE Api through Cloud requires connecting your service account through a JSON Key
# https://gis.stackexchange.com/questions/350527/authentication-issue-earth-engine-python-using-ee-serviceaccountcredentials

import ee
service_account = 'gee-spurge-dl-proj@my-account.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'jsonKeys/my-account-key.json')
ee.Initialize(credentials)

# Connect to to google cloud

import os
from google.cloud import storage
import gcloud
from google.oauth2 import service_account

# Set environment variables
# Set environment variable GOOGLE_APPLICATION_CREDENTIALS to the path to a service account credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'jsonKeys/my-account-key.json'

# Solves issue connecting SSL cert request to google cloud storage bucket
#https://stackoverflow.com/questions/63177156/tensorflow-dataloading-issue
os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"

SERVICE_ACCOUNT_FILE = 'jsonKeys/my-account-key.json'
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)


# Other module imports
# Using conda environment "earthengine"

import os
import pandas as pd
import numpy as np
import datetime
import pprint
import time
from functools import reduce
from pprint import pprint
import geemap #advanced python function for GEE
import fsspec # file system specification

# Tensorflow setup.

import tensorflow as tf
print(tf.__version__)


# Define a function to transfer feature properties to a dictionary.
def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)


#Cloud Mask: https://gis.stackexchange.com/questions/274048/apply-cloud-mask-to-landsat-imagery-in-google-earth-engine-python-api
def getQABits(image, start, end, mascara): 
    # Compute the bits we need to extract.
    pattern = 0
    for i in range(start,end+1):
        pattern += 2**i
    # Return a single band image of the extracted QA bits, giving the     band a new name.
    return image.select([0], [mascara]).bitwiseAnd(pattern).rightShift(start)


#Saturated band Mask: https://gis.stackexchange.com/questions/363929/how-to-apply-a-bitmask-for-radiometric-saturation-qa-in-a-image-collection-eart
def bitwiseExtract(value, fromBit, toBit):
  maskSize = ee.Number(1).add(toBit).subtract(fromBit)
  mask = ee.Number(1).leftShift(maskSize).subtract(1)
  return value.rightShift(fromBit).bitwiseAnd(mask)


#Function to mask out cloudy and saturated pixels and harmonize between Landsat 5/7/8 imagery 
def maskQuality(image):
    # Select the QA band.
    QA = image.select('QA_PIXEL')
    # Get the internal_cloud_algorithm_flag bit.
    sombra = getQABits(QA,3,3,'cloud_shadow') #shadow
    nubes = getQABits(QA,5,5,'cloud') #cloud
    #  var cloud_confidence = getQABits(QA,6,7,  'cloud_confidence')
    cirrus_detected = getQABits(QA,9,9,'cirrus_detected')
    #var cirrus_detected2 = getQABits(QA,8,8,  'cirrus_detected2')
    #Return an image masking out cloudy areas.
    QA_radsat = image.select('QA_RADSAT')
    saturated = bitwiseExtract(QA_radsat, 1, 7)

    #Apply the scaling factors to the appropriate bands.
    def getFactorImg(factorNames):
      factorList = image.toDictionary().select(factorNames).values()
      return ee.Image.constant(factorList)

    scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10'])

    offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10'])
    
    scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg)

    #Replace original bands with scaled bands and apply masks.
    return image.addBands(scaled, None, True).updateMask(sombra.eq(0)).updateMask(nubes.eq(0)).updateMask(cirrus_detected.eq(0).updateMask(saturated.eq(0))))


# Selects and renames bands of interest for Landsat OLI.
def renameOli(img):
  return img.select(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT'],
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL', 'QA_RADSAT'])


# Selects and renames bands of interest for Landsat TM/ETM+.
def renameEtm(img):
  return img.select(
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT'],
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL', 'QA_RADSAT'])


# Adding a NDVI band
def addNDVI(image):
  ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
  return image.addBands([ndvi])

#Get date of image
def mapDates(image):
  date = ee.Date(image.get('system:time_start')).format("YYYY-MM-dd")
  return image.addBands([date])

# Prepares (renames) OLI images.
def prepOli(img):
  img = renameOli(img)
  return img


# Prepares (renames) TM/ETM+ images.
def prepEtm(img):
  orig = img
  img = renameEtm(img)
  return ee.Image(img.copyProperties(orig, orig.propertyNames()))


# Selects and renames bands of interest for TM/ETM+.
def renameImageBands_TM(img, year, season):
  return img.select(
      ['Blue_median', 'Green_median', 'Red_median', 'NIR_median', 
       'SWIR1_median', 'SWIR2_median', 'NDVI_median'],
      ['Blue'+str(season)+str(year), 'Green'+str(season)+str(year), 'Red'+str(season)+str(year), 'NIR'+str(season)+str(year),
       'SWIR1'+str(season)+str(year), 'SWIR2'+str(season)+str(year), 'NDVI'+str(season)+str(year)])

# Selects and renames bands of interest for TM/ETM+.
def renameImageBands_ETMOLI(img, year, season):
  return img.select(
      ['Blue_median_median', 'Green_median_median', 'Red_median_median', 'NIR_median_median', 
       'SWIR1_median_median', 'SWIR2_median_median', 'NDVI_median_median'],
      ['Blue'+str(season)+str(year), 'Green'+str(season)+str(year), 'Red'+str(season)+str(year), 'NIR'+str(season)+str(year),
       'SWIR1'+str(season)+str(year), 'SWIR2'+str(season)+str(year), 'NDVI'+str(season)+str(year)])


def getLandsatMosaicFromPoints(year, points):
  '''
  #Time-series extraction developed from
  #https://developers.google.com/earth-engine/tutorials/community/time-series-visualization-with-altair#combine_dataframes  

  '''

  #if Year is between 1985 and 1999 use Landsat 5 TM imagery
  if 1985 <= year <= 1999:

    tmColMarchApril = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-03-01'.format(year), '{}-04-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColMarchApril = renameImageBands_TM(tmColMarchApril, year, 'MarchApril')

    tmColMayJune = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-05-01'.format(year), '{}-06-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColMayJune = renameImageBands_TM(tmColMayJune, year, 'MayJune')

    tmColJulyAug = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-07-01'.format(year), '{}-08-31'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColJulyAug = renameImageBands_TM(tmColJulyAug, year, 'JulyAug')

    landsat5ImageCol = [tmColMarchApril, tmColMayJune, tmColJulyAug]
    return landsat5ImageCol

  #if Year is between 2000 and 2012 use mosaic from Landsat 5 TM and Landsat 7 ETM imagery
  elif 2000 <= year <= 2012:

    etmColMarchApril = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-03-01'.format(year), '{}-04-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColMarchApril = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-03-01'.format(year), '{}-04-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    MarchApril = ee.ImageCollection([etmColMarchApril, tmColMarchApril])

    etmColMarchApril = MarchApril.reduce('median')

    etmColMarchApril = renameImageBands_ETMOLI(etmColMarchApril, year, 'MarchApril')

    etmColMayJune = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-05-01'.format(year), '{}-06-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColMayJune = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-05-01'.format(year), '{}-06-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    MayJune = ee.ImageCollection([etmColMayJune, tmColMayJune])

    etmColMayJune = MayJune.reduce('median')

    etmColMayJune = renameImageBands_ETMOLI(etmColMayJune, year, 'MayJune')

    etmColJulyAug = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-07-01'.format(year), '{}-08-31'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    tmColJulyAug = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
      .filterDate('{}-07-01'.format(year), '{}-08-31'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    JulyAug = ee.ImageCollection([etmColJulyAug, tmColJulyAug])

    etmColJulyAug = JulyAug.reduce('median')

    etmColJulyAug = renameImageBands_ETMOLI(etmColJulyAug, year, 'JulyAug')

    landsat5_7ImageCol = [etmColMarchApril, etmColMayJune, etmColJulyAug]
    return landsat5_7ImageCol

  #if Year is between 2013 and 2020 use mosaic from Landsat 7 ETM and Landsat 8 OLI imagery
  elif 2013 <= year <= 2023:

    etmColMarchApril = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-03-01'.format(year), '{}-04-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    oliColMarchApril = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
      .filterDate('{}-03-01'.format(year), '{}-04-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepOli) \
      .map(addNDVI) \
      .reduce('median')

    MarchApril = ee.ImageCollection([etmColMarchApril, oliColMarchApril])

    etmColMarchApril = MarchApril.reduce('median')

    etmColMarchApril = renameImageBands_ETMOLI(etmColMarchApril, year, 'MarchApril')

    etmColMayJune = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-05-01'.format(year), '{}-06-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median')

    oliColMayJune = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
      .filterDate('{}-05-01'.format(year), '{}-06-30'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepOli) \
      .map(addNDVI) \
      .reduce('median')

    MayJune = ee.ImageCollection([etmColMayJune, oliColMayJune])

    etmColMayJune = MayJune.reduce('median')

    etmColMayJune = renameImageBands_ETMOLI(etmColMayJune, year, 'MayJune')

    etmColJulyAug = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
      .filterDate('{}-07-01'.format(year), '{}-08-31'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepEtm) \
      .map(addNDVI) \
      .reduce('median') \

    oliColJulyAug = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
      .filterDate('{}-07-01'.format(year), '{}-08-31'.format(year)) \
      .filterBounds(points) \
      .map(maskQuality) \
      .map(prepOli) \
      .map(addNDVI) \
      .reduce('median')

    JulyAug = ee.ImageCollection([etmColJulyAug, oliColJulyAug])

    etmColJulyAug = JulyAug.reduce('median')

    etmColJulyAug = renameImageBands_ETMOLI(etmColJulyAug, year, 'JulyAug')

    landsat7_8ImageCol = [etmColMarchApril, etmColMayJune, etmColJulyAug]

    return landsat7_8ImageCol



def sampleImagestoDataFrame(listofEEImages):
    '''
    Function takes in a list of three images from a Landsat imagery year (T1, T2, T3)
    Returns a merged pandas dataframe of dimensions (rows/samples x bands) ordered from t-1, t, t+1
    '''
    image1 = listofEEImages[0]
    image2 = listofEEImages[1]
    image3 = listofEEImages[2]

    image1_fc = image1.sampleRegions(collection=newpts, properties=['class'], scale=30)
    image2_fc = image2.sampleRegions(collection=newpts, properties=['class'], scale=30)
    image3_fc = image3.sampleRegions(collection=newpts, properties=['class'], scale=30)

    image1_db_dict = fc_to_dict(image1_fc).getInfo()
    image2_db_dict = fc_to_dict(image2_fc).getInfo()
    image3_db_dict = fc_to_dict(image3_fc).getInfo()

    image1_df = pd.DataFrame(image1_db_dict)
    image2_df = pd.DataFrame(image2_db_dict)
    image3_df = pd.DataFrame(image3_db_dict)

    data_frames = [image1_df, image2_df, image3_df]

    df_merged = reduce(lambda left,right: pd.merge(left, right, on='system:index', how='outer'), data_frames).fillna(np.nan)

    df_merged_dropna = df_merged.dropna(axis=0, how = 'any')

    return df_merged_dropna



# Our study area is rather large, so we break it up into several areas with a sliding window
# In each of these areas, we will select points to extract Landsat imagery & NLCD class values
# Generate Bounding Box Coordinate List for Study Region ###

#Generate Bounding Box Coordinate List for Study Region ###
#Starting position of bounding box
XY_topLeft = [-116.976099, 48.904682]
XY_topRight = [-115.976099, 48.904682]
XY_bottomLeft = [-116.976099, 47.904682]
XY_bottomRight = [-115.976099, 47.904682]

lon_range = 31 #study area spans 31 deg lon
lat_range = 13 #study area spans 12 deg lat

stepSize = 1 #step by 1 degree of long/latitude

def sliding_window(longitude_range, latitude_range, stepSize_box):
    lon_list = []
    lat_list = []
    for lon in range(0, longitude_range, stepSize_box):
      for lat in range(0, latitude_range,stepSize_box):
        lon_list.append(lon)
        lat_list.append(lat)
    
    return(lon_list, lat_list)

def bbox(longitude_range, latitude_range, stepSize_box, topLeft_coord, topRight_coord, bottomLeft_coord, bottomRight_coord, origin_coord):
  #Creates a sliding window across the lat/long range
  #Returns a list of all lat/long boxes to sample 
     
  lon_list, lat_list = sliding_window(longitude_range, latitude_range, stepSize_box) #Generates two lists: one of longitude[0-31] and one of latitude [0-12] defining study region

  #Top Left Coordinates for BBox
  lon_list_X_topLeft = [x + topLeft_coord[0] for x in lon_list]
  lat_list_Y_topLeft = [abs(x - topLeft_coord[1]) for x in lat_list]
  XY_topLeft_list = list(zip(lon_list_X_topLeft, lat_list_Y_topLeft))

  #Bottom Left Coordinates for BBox
  lon_list_X_bottomLeft = [x + bottomLeft_coord[0] for x in lon_list]
  lat_list_Y_bottomLeft = [abs(x - bottomLeft_coord[1]) for x in lat_list]
  XY_bottomLeft_list = list(zip(lon_list_X_bottomLeft, lat_list_Y_bottomLeft))

  #Top Right Coordinates for BBox
  lon_list_X_topRight = [x + topRight_coord[0] for x in lon_list]
  lat_list_Y_topRight = [abs(x - topRight_coord[1]) for x in lat_list]
  XY_topRight_list = list(zip(lon_list_X_topRight, lat_list_Y_topRight))

  #Bottom Right Coordinates for BBox
  lon_list_X_bottomRight = [x + bottomRight_coord[0] for x in lon_list]
  lat_list_Y_bottomRight = [abs(x - bottomRight_coord[1]) for x in lat_list]
  XY_bottomRight_list = list(zip(lon_list_X_bottomRight, lat_list_Y_bottomRight))

  ### Bounding Box Coordinate List
  bbox = list(zip(XY_topLeft_list, XY_topRight_list, XY_bottomLeft_list, XY_bottomRight_list, XY_topLeft_list))

  return bbox


bbox_windows = bbox(lon_range, lat_range, stepSize, XY_topLeft, XY_topRight, XY_bottomRight, XY_bottomLeft, XY_topLeft)


print(bbox_windows[input_value])


# Define export for feature class assets
OUTPUT_BUCKET = 'spurge-demography-geotiffs'

# Make sure the bucket exists.
print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + OUTPUT_BUCKET) 
  else 'Output Cloud Storage bucket does not exist.')


TRAIN_FILE_PREFIX = 'Training_nlcd1986_image'
TEST_FILE_PREFIX = 'Testing_nlcd1986_image'
VALID_FILE_PREFIX = 'Validation_nlcd1986_image'

#file_extension = '.tfrecord.gz'

# File name for the prediction (image) dataset.  The trained model will read
# this dataset and make predictions in each pixel.
IMAGE_FILE_PREFIX = 'landsat_image_temporalcnn_1986_tile'


BANDS = ['0_BlueMarchApril1985',
 '0_GreenMarchApril1985',
 '0_RedMarchApril1985',
 '0_NIRMarchApril1985',
 '0_SWIR1MarchApril1985',
 '0_SWIR2MarchApril1985',
 '0_NDVIMarchApril1985',
 '0_BlueMayJune1985',
 '0_GreenMayJune1985',
 '0_RedMayJune1985',
 '0_NIRMayJune1985',
 '0_SWIR1MayJune1985',
 '0_SWIR2MayJune1985',
 '0_NDVIMayJune1985',
 '0_BlueJulyAug1985',
 '0_GreenJulyAug1985',
 '0_RedJulyAug1985',
 '0_NIRJulyAug1985',
 '0_SWIR1JulyAug1985',
 '0_SWIR2JulyAug1985',
 '0_NDVIJulyAug1985',
 '1_BlueMarchApril1986',
 '1_GreenMarchApril1986',
 '1_RedMarchApril1986',
 '1_NIRMarchApril1986',
 '1_SWIR1MarchApril1986',
 '1_SWIR2MarchApril1986',
 '1_NDVIMarchApril1986',
 '1_BlueMayJune1986',
 '1_GreenMayJune1986',
 '1_RedMayJune1986',
 '1_NIRMayJune1986',
 '1_SWIR1MayJune1986',
 '1_SWIR2MayJune1986',
 '1_NDVIMayJune1986',
 '1_BlueJulyAug1986',
 '1_GreenJulyAug1986',
 '1_RedJulyAug1986',
 '1_NIRJulyAug1986',
 '1_SWIR1JulyAug1986',
 '1_SWIR2JulyAug1986',
 '1_NDVIJulyAug1986',
 '2_BlueMarchApril1987',
 '2_GreenMarchApril1987',
 '2_RedMarchApril1987',
 '2_NIRMarchApril1987',
 '2_SWIR1MarchApril1987',
 '2_SWIR2MarchApril1987',
 '2_NDVIMarchApril1987',
 '2_BlueMayJune1987',
 '2_GreenMayJune1987',
 '2_RedMayJune1987',
 '2_NIRMayJune1987',
 '2_SWIR1MayJune1987',
 '2_SWIR2MayJune1987',
 '2_NDVIMayJune1987',
 '2_BlueJulyAug1987',
 '2_GreenJulyAug1987',
 '2_RedJulyAug1987',
 '2_NIRJulyAug1987',
 '2_SWIR1JulyAug1987',
 '2_SWIR2JulyAug1987',
 '2_NDVIJulyAug1987']

LABEL = 'class'

# Number of label values, i.e. number of classes in the classification.
N_CLASSES = 10

# These names are used to specify properties in the export of
# training/testing data and to define the mapping between names and data
# when reading into TensorFlow datasets.
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append(LABEL)


# Define years to sample data (corresponds to satellite image year)
years = [2021, 2022, 2023]

#Training points for leafy spurge & land cover classes (defines extent of landsat imagery)

#Load 1m training points sampled from 2019 NLCD and leafy spurge from 2018-2019-2020
pts = ee.FeatureCollection('projects/ee-rbriscoe/assets/spurge_landcover_nlcd2019_onemillionpts_sept2022')

print('Tile ' + str(input_value))

# Define Bounding Box
bbox = bbox_windows[input_value]
print(bbox)

# Filter points based on AOI
aoi = ee.Geometry.Polygon(bbox)

#Apply Filter
newpts = pts.filterBounds(aoi)

#How many points?
count = newpts.size() #returns an EE.Number object that we need to convert to an interger
num_points = int(count.getInfo())
print('Number of Points within AOI (Count): ', str(count.getInfo())+'\n')

if num_points > 0:
    # Sample imagery in a year filtered by input points
    # Output is a list of length 3 EEimages, corresponding to three seasons in a year (e.g 2018: MarchApril, MayJune, JulyAug)
    LandsatCol_year0 = getLandsatMosaicFromPoints(years[0], newpts)

    LandsatCol_year1 = getLandsatMosaicFromPoints(years[1], newpts)

    LandsatCol_year2 = getLandsatMosaicFromPoints(years[2], newpts)

    LandsatCol_timeseries = ee.ImageCollection([LandsatCol_year0, LandsatCol_year1, LandsatCol_year2])

    LandsatCol_timeseries_image = LandsatCol_timeseries.toBands().multiply(10000)
    #LandsatCol_timeseries_image.bandNames().getInfo()

     # Export imagery in this region.
    EXPORT_REGION = aoi

    # Specify patch and file dimensions.
    #image_export_options = {
    #  'patchDimensions': [512, 512],
    #  'maxFileSize': 104857600,
    #  'compressed': True
    #}

    # Setup the task.
    image_task = ee.batch.Export.image.toCloudStorage(
      image=LandsatCol_timeseries_image.toInt16(),
      description='Image Export',
      bucket=OUTPUT_BUCKET,
      fileNamePrefix=IMAGE_FILE_PREFIX + "_" + str(input_value) + "_",
      region=EXPORT_REGION.toGeoJSON()['coordinates'],
      scale=30,
      fileFormat='GeoTIFF',
      maxPixels = 1e12
    )

    # Start the task.
    image_task.start() #takes around 20 minutes?

    #Wait for export tasks to finish
    while image_task.active():
      print('Polling for image task (state: {}).'.format(ee.data.getTaskStatus(image_task.id)[0].get('state')))
      time.sleep(120)


print("Image Task Export Finished")        
        
#EOF