# Code taken from Sentdex's  Preprocessing tutorial found here: https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet/notebook
# Initial tutorial was a large ipython notebook. We have condesed this to a single python script and made it usable for our application. 

import numpy as np
import pandas as pd
import dicom
import os
import cv2
import math
import sys


raw_data = []

train = False

IMG_SIZE_PX = 50
SLICE_COUNT = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(a):
    return sum(a) / len(a)

#-----------------------------------------------------------------------------------------
# process_dataTrain
# IN: patient - patient ID
#     labels_df - data value for the cancer prognosis
# OUT: Numpy array of the slices from patient file,
#      updated patient label with cancer prognosis appended for training purposes
#
# FUNCTION: Takes the raw dicom image files from each patient folder, sorts them 
# for processing and combinition for 3D array output. Function also resizes image files
# to 50 x 50 pixel images. Function used when processing images for training of Neural Net
#-----------------------------------------------------------------------------------------	
def process_dataTrain(patient,labels_df,img_px_size=50, hm_slices=20):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    print('%s' % path)
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
        
    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
 
    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(new_slices),label

#-----------------------------------------------------------------------------------------
# process_dataNorm
# IN: patient - patient ID
# OUT: Numpy array of the slices from patient file, patient ID used for label
#
# FUNCTION: Takes the raw dicom image files from each patient folder, sorts them 
# for processing and combinition for 3D array output. Function also resizes image files
# to 50 x 50 pixel images. Function used when processing images for non-training analysis
#-----------------------------------------------------------------------------------------
def process_dataNorm(patient,img_px_size=50, hm_slices=20):
    
    path = data_dir + patient
    print('%s' % path)
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
        
    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
 
    return np.array(new_slices)

argument = sys.argv[1]

if argument == 'train':
    data_dir = 'training/'
    labels = pd.read_csv('train_data.csv', index_col=0)
    train = True
else:
    data_dir = 'patients/'
    labels = pd.read_csv('patient_data.csv', index_col=0)
    train = False

patients = os.listdir(data_dir)

for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        if train:
           img_data,label = process_dataTrain(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
           print(img_data.shape,label)
           raw_data.append([img_data,label])
        else:
           img_data = process_dataNorm(patient,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
           print(img_data.shape)
           raw_data.append([img_data])
    except KeyError as e:
        print('This is unlabeled data!')


if train:
    np.save('traindata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), raw_data)
else:
    np.save('patientdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), raw_data)






