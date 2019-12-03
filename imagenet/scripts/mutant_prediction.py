# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:19:43 2019

@author: fwursd
"""

import os

import pickle
import numpy as np
from imagenet_utils import *
from keras.preprocessing.image import ImageDataGenerator


parser = argparse.ArgumentParser(description='parser to get the value of num_class')
parser.add_argument("--num_class", default = 100, help = 'can be either 50 or 100')
args = parser.parse_args()

num_class = args.num_class

model = create_net('model/reduced_weights{}.h5'.format(num_class), num_class)

paths = [
    '/data/ImageNet/train{}/'.format(num_class),
    '/data/ImageNet/val{}/'.format(num_class),
    '/data/ImageNet/test{}/'.format(num_class)]
    
paths_string = ['train100', 'val100', 'test100']
root_dir = 'mutant_predictions'


if __name__ == '__main__':

    list_jobs = [
        # ('GF', 0.002),
        ('GF', 0.0025),
        # ('NAI', 0.0015),
        # ('GF', 0.0015),
        ('NAI', 0.0025),
    ]

    for method, rate in list_jobs:

        for p in range(len(paths)):

            datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
            gen = datagen.flow_from_directory(paths[p],target_size=(299,299),
                class_mode='categorical',shuffle=False,batch_size=124)

            print(paths_string[p])
            path = os.path.join(root_dir, f'{method}_{rate}', paths_string[p])
            print(path)
            os.makedirs(path, exist_ok=True)
            for i in range(1):
                print(i)
                mutations = pickle.load(open(f'{i}_100Class_{method}_mutations_{rate}.p', 'rb'))
                for n in range(len(mutations)):
                    print("---- n="+str(n)+" ----", end = '\r')
                    model.set_weights(mutations[n])
                    gen.reset()
                    mutated_pred = model.predict_generator(gen, steps = gen.n // gen.batch_size + 1, verbose = 1)
                    mutated_pred = np.argmax(mutated_pred, axis = 1)

                    filename = os.path.join(path, f'{i}_mutant{n}_0.002.p')
                    pickle.dump(mutated_pred, open(filename, 'wb'))
                print()
