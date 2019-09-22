from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.models import Model
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
import pickle
from imagenet_utils import *

path = '/data/Datasets/ImageNet/train100/'
path = 'train_small/'
classes = sorted(os.listdir(path))
num_class = len(classes)

net = InceptionV3(weights = None, include_top = False, input_shape = (299,299,3))
x = net.output
x = GlobalAveragePooling2D()(x)
x = Dense(num_class)(x)
outputs = Activation('softmax')(x)
model = Model(inputs = net.input, outputs = outputs)
model.load_weights('reduced_weights.h5')

convLayers = []
for i in range(len(model.layers)):
    if ("mixed" in model.layers[i].name) & (model.layers[i].name[-2] != '_'):
        convLayers.append(model.layers[i])
convBlocks = getBlocks(convLayers)

train_pred = pickle.load(open('train_pred100.p', 'rb'))

if __name__ == '__main__':
    sess = K.get_session()
    input_tensor = model.layers[0].input
    output_tensor = model.layers[-2].output
    for k in reversed(range(len(convBlocks))):
        inter_tensors = [conv.output for conv in convBlocks[k]]
        outputs = [tf.gradients(output_tensor[:,i],inter_tensors) for i in range(num_class)]
        outputs.append(inter_tensors)
        rights = {}
        rights_ind = {}
        wrongs = {}
        wrongs_ind = {}
        for c in range(num_class):
            right = []
            right_ind = []
            for i in range(len(os.listdir(path+classes[c]))):
              
                array = load_array(c,i,path)
                pred = int(train_pred[c][i])
                
                back = "\033[F\033[K"
                print(back + f"class {c + 1:3d}/{num_class}, image {i + 1:4d}/{len(os.listdir(path+classes[c]))}")

                grads, activs = integratedGrad(array,input_tensor, [outputs[pred], outputs[-1]], sess)
                grads, activs = np.array(grads), np.array(activs)
                delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
                contribs = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
                contribs = np.mean(contribs, axis = (0,2,3))
                        
                if pred == c:
                    right.append(contribs)
                    right_ind.append(i)
                else:
                    if pred not in wrongs.keys():
                        wrongs[pred] = []
                        wrongs_ind[pred] = []
                        wrongs_ind[pred].append((c,i))
                        wrongs[pred].append(contribs)
                    else:
                        wrongs_ind[pred].append((c,i))
                        wrongs[pred].append(contribs)
            rights[c] = right
            rights_ind[c] = right_ind

        pickle.dump(rights, open('rights100_block'+str(k)+'.p', 'wb'))
        pickle.dump(rights_ind, open('rights100_ind.p', 'wb'))
        pickle.dump(wrongs, open('wrongs100_block'+str(k)+'.p', 'wb'))
        pickle.dump(wrongs_ind, open('wrongs100_ind.p', 'wb'))
