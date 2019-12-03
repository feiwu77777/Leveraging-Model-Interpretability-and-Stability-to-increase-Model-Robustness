from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.models import Model
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
import pickle
import argparse

from imagenet_utils import *


parser = argparse.ArgumentParser(description='parser to get the value of num_class')
parser.add_argument("--num_class", default = 100, help = 'is either 50 or 100')
args = parser.parse_args()

num_class = args.num_class

path = 'Datasets/train{}/'.format(num_class) #path to the dataset of 100 classes Imagenet
classes = sorted(os.listdir(path))

#the model here is already trained on the 100/50 classes Imagenet, see 'train_reduced_model.py'
model = create_net('model/reduced_weights{}.h5'.format(num_class), num_class)

#get all conv layer in the network, for inception, only the output of mixed layers are considered
convLayers = get_convLayers(model)
#group conv layers by number of channel
convBlocks = getBlocks(convLayers) 


#make prediction on the training set before hand and load them here to save time
train_pred = pickle.load(open('train_pred{}.p'.format(num_class), 'rb'))

if __name__ == '__main__':

    allConvBlock_wrongs = []
    allConvBlock_rights = []

    for block in convBlocks:
        #input_tensor is the input tensor given to the model
        #outputs are gradients and activations associated to the output of block 
        input_tensor, outputs, sess = init(block, model, num_class) 
        cond_rights = {}
        cond_wrongs = {}
        originIndex_rights = {}
        originIndex_wrongs = {}

        for c in range(num_class):

            cond_right = []
            right_ind = []

            #iterating over all wrong/correct prediction to calculate their conductance
            for i in range(len(os.listdir(path+classes[c]))):
              
                array = load_array(c,i,path)
                pred = int(train_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(path+classes[c]))}")

                #calculate the gradient of the predicted index wrt to the activation output of block
                grads, activs = integratedGrad(array,input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)

                #calculate the conductance for neurons/output of block
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                cond = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                cond = np.mean(cond, axis = (0,2,3))
                        
                #in the case of a correct prediction
                if pred == c:
                    cond_right.append(cond)
                    right_ind.append(i)

                #in the case of a wrong prediction, store conductance grouped by the class in which the image is predicted
                #originIndex store the original index of the image so its true label can be found
                else:
                    if pred not in cond_wrongs.keys():
                        cond_wrongs[pred] = []
                        cond_wrongs[pred].append(cond)
                        originIndex_wrongs[pred] = []
                        originIndex_wrongs[pred].append((c,i))
                    else:
                        originIndex_wrongs[pred].append((c,i))
                        cond_wrongs[pred].append(cond)
            cond_rights[c] = cond_right
            originIndex_rights[c] = right_ind

        allBlockCond_wrongs.append(cond_wrongs)
        allBlockCond_rights.append(cond_rights)

        #reformatting the conductances of all layers
        cond_wrongs = allBlockCond_wrongs[0]
        cond_rights = allBlockCond_rights[0]
        for c in range(num_class):
            for i in range(len(cond_wrongs[c])):
                cond_wrongs[c][i] = list(cond_wrongs[c][i])
                for b in range(1, len(convBlocks)):
                    cond_wrongs[c][i].extend(list(allBlockCond_wrongs[b][c][i]))
            for i in range(len(cond_rights[c])):
                cond_rights[c][i] = list(cond_rights[c][i])
                for b in range(1, len(convBlocks)):
                    cond_rights[c][i].extend(list(allBlockCond_rights[b][c][i]))

    pickle.dump(cond_rights, open('pickled/cond_rights{}.p'.format(num_class), 'wb'))
    pickle.dump(cond_wrongs, open('pickled/cond_wrongs{}.p'.format(num_class), 'wb'))
    pickle.dump(originIndex_rights, open('pickled/originIndex_rights{}.p'.format(num_class), 'wb'))
    pickle.dump(originIndex_wrongs, open('pickled/originIndex_wrongs{}.p'.format(num_class), 'wb'))
