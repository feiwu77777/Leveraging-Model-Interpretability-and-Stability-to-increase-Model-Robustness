# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:14 2019

@author: fwursd
"""

import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
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

num_class = 50
model = create_net('reduced_weights.h5', num_class)

path = '/data/Datasets/ImageNet/val_small/'
#path = 'val_small/'


bs = 100
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
val_gen = val_datagen.flow_from_directory(path,target_size=(299,299),class_mode='categorical',shuffle=False,batch_size=bs)
opt = keras.optimizers.Adam(1e-4)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
#evaluation = model.evaluate_generator(val_gen, steps = val_gen.n // val_gen.batch_size, verbose = 1)
#val_acc = evaluation[1]
val_acc = 0.9812

params_num, allWeights = process_model_params(model)
        
if __name__ == '__main__':
    GF_mutations = create_mutations('GF',model, params_num, 
                                       allWeights, 'reduced_weights.h5', 
                                       0.002, val_acc, val_gen, limit = 100)
    pickle.dump(GF_mutations, open('GF_mutations.p', 'wb'))