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
import argparse

from imagenet_utils import *


parser = argparse.ArgumentParser(description='parser to get the value of num_class')
parser.add_argument("--num_class", default = 100, help = 'can be either 50 or 100')
args = parser.parse_args()

num_class = args.num_class

home = '../../'
val_path = 'Datasets/val{}/'.format(num_class) #path to the dataset of 100 classes Imagenet
test_path = 'Datasets/test{}/'.format(num_class) #path to the dataset of 100 classes Imagenet
val_classes = sorted(os.listdir(val_path))
test_classes = sorted(os.listdir(test_path))


#the model here is already trained on the 100/50 classes Imagenet, see 'train_reduced_model.py'
model = create_net('model/reduced_weights{}.h5'.format(num_class), num_class)

#get all conv layer in the network, for inception, only the output of mixed layers are considered
convLayers = get_convLayers(model)
#group conv layers by number of channel
convBlocks = getBlocks(convLayers)

#make prediction on the dataset before hand and load them here to save time    
val_pred = pickle.load(open('pickled/val{}_pred.p'.format(num_class), 'rb'))
test_pred = pickle.load(open('pickled/test{}_pred.p'.format(num_class), 'rb'))

if __name__ == '__main__':

    allConvBlock_val = []
    allConvBlock_test = []

    for block in convBlocks:
        #input_tensor is the input tensor given to the model
        #outputs are gradients and activations associated to the output of block 
        input_tensor, outputs, sess = init(convBlocks[k], model, num_class)

        cond_val = {}
        cond_test = {}
        originIndex_val = {}
        originIndex_test = {}
 
        for c in range(num_class):

            #iterating over all wrong/correct prediction of val set to calculate their conductance
            for i in range(len(os.listdir(val_classes[c]))):
                array = load_array(c,i,val_path)
                pred = int(val_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(val_classes[c]))}")

                #calculate the gradient of the predicted index wrt to the activation output of block
                grads, activs = integratedGrad(array, input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)

                #calculate the conductance for neurons/outputs of block
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                cond = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                cond = np.mean(cond, axis = (0,2,3))

                #store conductance grouped by the class in which the image is predicted
                #originIndex store the original index of the image so its true label can be found
                if pred not in cond_val.keys():
                    cond_val[pred] = []
                    cond_val[pred].append(cond)
                    originIndex_val[pred] = []
                    originIndex_val[pred].append((c,i))
                else:
                    cond_val[pred].append(cond)
                    originIndex_val[pred].append((c,i))

            #iterating over all wrong/correct prediction of test set to calculate their conductance
            for i in range(len(os.listdir(test_classes[c]))):
                array = load_array(c,i,test_path)
                pred = int(test_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(test_classes[c]))}")

                #calculate the gradient of the predicted index wrt to the activation output of block               
                grads, activs = integratedGrad(array, input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)

                #calculate the conductance for neurons/outputs of block
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                cond = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                cond = np.mean(cond, axis = (0,2,3))

                #store conductance grouped by the class in which the image is predicted
                #originIndex store the original index of the image so its true label can be found               
                if pred not in cond_test.keys():
                    cond_test[pred] = []
                    cond_test[pred].append(cond)
                    originIndex_test[pred] = []
                    originIndex_test[pred].append((c,i))
                else:
                    cond_test[pred].append(cond)
                    originIndex_test[pred].append((c,i))

        allBlockCond_val.append(cond_val)
        allBlockCond_test.append(cond_test)

        #reformatting the conductances of all layers
        cond_val = allBlockCond_val[0]
        cond_test = allBlockCond_test[0]
        for c in range(num_class):
            for i in range(len(cond_val[c])):
                cond_val[c][i] = list(cond_val[c][i])
                for b in range(1, len(convBlocks)):
                    cond_val[c][i].extend(list(allBlockCond_val[b][c][i]))
            for i in range(len(cond_test[c])):
                cond_test[c][i] = list(cond_test[c][i])
                for b in range(1, len(convBlocks)):
                    cond_test[c][i].extend(list(allBlockCond_test[b][c][i]))
    

    pickle.dump(cond_val, open(home + 'pickled/cond_val{}.p'.format(num_class), 'wb'))
    pickle.dump(cond_test, open(home + 'pickled/cond_test{}.p'.format(num_class), 'wb'))
    pickle.dump(originIndex_val, open(home + 'pickled/originIndex_val{}.p'.format(num_class), 'wb'))
    pickle.dump(originIndex_test, open(home + 'pickled/originIndex_test{}.p'.format(num_class), 'wb'))