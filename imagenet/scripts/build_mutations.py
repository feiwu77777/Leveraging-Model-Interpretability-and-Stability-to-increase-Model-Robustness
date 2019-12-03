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
import argparse

from imagenet_utils import *


parser = argparse.ArgumentParser(description='parser')
parser.add_argument("--num_class", type = int, default = 100, help = 'is either 50 or 100')
parser.add_argument("--limit", type = int, default = 100, help = 'number of modified model')
parser.add_argument("--gamma", type = float, default = 0.002, help = 'percentage of weights to modify in the model')
parser.add_argument("--mode", type = string, default = 'NAI', help = 'define the type of operation\
                                                                     to apply for modifying weights\
                                                                     of the model, is either NAI or GF')
args = parser.parse_args()

num_class = args.class_num
mode = args.mode
gamma = args.gamma

model = create_net('model/reduced_weights{}.h5'.format(num_class), num_class)

path = 'Datasets/val{}/'.format(num_class)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
val_gen = val_datagen.flow_from_directory(path,target_size=(299,299),class_mode='categorical',shuffle=False,batch_size=100)
opt = keras.optimizers.Adam(1e-4)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
evaluation = model.evaluate_generator(val_gen, steps = val_gen.n // val_gen.batch_size, verbose = 1)
val_acc = evaluation[1]
#val_acc = 0.9812

        
if __name__ == '__main__':
    #get total number of parameter in the model and the position of all weights
    params_num, allWeights = process_model_params(model)

    mutations = []
    count = 0
    while len(mutations) != args.limit:
        print(count, end = "\r")
        count += 1
        model.load_weights('model/reduced_weights{}.h5'.format(num_class))

        #get random index of weights that will be changed
        rnd = np.arange(params_num)
        np.random.shuffle(rnd)
        rnd = rnd[:int(params_num*gamma)]
        rnd = sorted(rnd)

        #go over the weights and change their value
        for i in range(len(allWeights)):
            for num in rnd:
                if num in allWeights[i][1]:
                    index = np.argwhere(allWeights[i][1] == num).item()
                    w = model.layers[allWeights[i][0]].get_weights()[0]

                    #weights changes using Gaussian Fuzz
                    if mode == 'GF':
                        avg_w = np.mean(w, axis = -1)
                        std_w = np.std(w, axis = -1)
                        w[:,:,:,index] = np.random.normal(avg_w, std_w)
                    
                    #weights changes using Neuron Activation Inverse
                    if mode == 'NAI':
                        w[:,:,:,index] = -1*w[:,:,:,index]

                    model.layers[allWeights[i][0]].set_weights([w])

        #if the validation accuracy of the mutant is at least 90% of the orginal model's keep the mutant
        acc = model.evaluate_generator(val_gen, steps = val_gen.n // val_gen.batch_size, verbose = 1)[1]
        if acc > 0.9*val_acc:
            mutations.append(model.get_weights())

    pickle.dump(mutations, open('pickled/{}_mutations{}_{}.p'.format(mode, num_class, gamma), 'wb'))