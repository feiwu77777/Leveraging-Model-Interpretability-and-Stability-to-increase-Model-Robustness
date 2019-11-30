# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:01:26 2019

@author: fwursd
"""

import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, BatchNormalization
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential
from keras.datasets import cifar10
import numpy as np
import os
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from keras.preprocessing import image
from utils import *

#loading dataset
num_class = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
input_shape = x_train[0].shape

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

#loading model
model = resnet_v1(input_shape=input_shape, depth = 3*6+2)
model.load_weights('cifar10resnet_weights.h5')

test_pred = np.argmax(model.predict(x_test), axis = 1)
test_label =np.argmax(y_test, axis = 1)
test_acc = np.mean(test_pred == test_label)

params_num, allWeights = process_model_params(model)


if __name__ == '__main__':
    #mutations is a list that contains the weights of every mutant
    mutations = create_mutations('GF', model, params_num, allWeights, 
                                 'cifar10resnet_weights.h5', 0.015, 
                                 test_acc, x_test, test_label, limit = 100)
    pickle.dump(mutations, open('mutations_GF_0.015_CIFAR.p', 'wb'))