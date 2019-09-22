# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:27:18 2019

@author: fwursd
"""


import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten, Dropout, Dense, GlobalAveragePooling2D, Activation
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import pickle
import os
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from imagenet_utils import *


val_path = '/data/Datasets/ImageNet/val100/'
test_path = '/data/Datasets/ImageNet/test100/'
val_classes = sorted(os.listdir(val_path))
test_classes = sorted(os.listdir(test_path))
num_class = 100

model = create_net('reduced_weights.h5', num_class)

convLayers = []
for i in range(len(model.layers)):
    if ("mixed" in model.layers[i].name) & (model.layers[i].name[-2] != '_'):
        convLayers.append(model.layers[i])
convBlocks = getBlocks(convLayers)
            
val_pred = pickle.load(open('val100_pred.p', 'rb'))
test_pred = pickle.load(open('test100_pred.p', 'rb'))

if __name__ == '__main__':
    for k in reversed(range(len(convBlocks))):
        input_tensor, outputs, sess = init(convBlocks[k], model, num_class)
        val_cond = {}
        val_index = {}
        test_cond = {}
        test_index = {}
        for c in range(num_class):

            for i in range(len(os.listdir(val_classes[c]))):
                array = load_array(c,i,val_path)
                pred = int(val_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(val_classes[c]))}")
                
                grads, activs = integratedGrad(array, input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                contribs = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                contribs = np.mean(contribs, axis = (0,2,3))
                
                if pred not in val_index.keys():
                    val_cond[pred] = []
                    val_cond[pred].append(contribs)
                    val_index[pred] = []
                    val_index[pred].append((c,i))
                else:
                    val_cond[pred].append(contribs)
                    val_index[pred].append((c,i))

            for i in range(len(os.listdir(test_classes[c]))):
                array = load_array(c,i,test_path)
                pred = int(test_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(test_classes[c]))}")
                
                grads, activs = integratedGrad(array, input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                contribs = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                contribs = np.mean(contribs, axis = (0,2,3))
                
                if pred not in test_index.keys():
                    test_cond[pred] = []
                    test_cond[pred].append(contribs)
                    test_index[pred] = []
                    test_index[pred].append((c,i))
                else:
                    test_cond[pred].append(contribs)
                    test_index[pred].append((c,i))
                    
        pickle.dump(val_cond, open('val100_cond_block'+str(k)+'.p', 'wb'))
        pickle.dump(val_index, open('val100_index.p', 'wb'))
        pickle.dump(val_cond, open('test100_cond_block'+str(k)+'.p', 'wb'))
        pickle.dump(val_index, open('test100_index.p', 'wb'))