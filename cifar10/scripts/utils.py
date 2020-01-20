# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:57:37 2019

@author: fwursd
"""
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, BatchNormalization
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
import keras
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


def resnet_layer(inputs, num_filters: int = 16, kernel_size: int = 3, strides: int = 1, 
                 activation: str = 'relu', batch_normalization: bool = True, conv_first: bool = True):
    """
    Function: helper to the function resnet_v1
    Inputs: layer characteristics
    Outputs: a resnet layer
    """
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', 
                  kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape: tuple, depth: int, num_classes: int = 10):
    """
    Function: construct a resnet with specified parameters 
    Inputs: characteristics of the neural network
    Output: the specified neural network
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



def initialization(model, num_class: int,  inter_layers: list, output_activ: bool):
    """
    function: initialize graph variable to calculate gradients and activations
    inputs: a model
    outputs: initialized graph for activations inside the model
    """
    sess = K.get_session()
    input_tensor = model.layers[0].input
    inter_tensors = [layer.output for layer in inter_layers]
    if output_activ == True:
        out = model.layers[-2].output
        weights, bias = model.layers[-1].weights
        output_tensor = tf.add(tf.matmul(out,weights),bias)
    else:
        output_tensor = model.layers[-2].output
    gradients = [tf.gradients(output_tensor[:,i],inter_tensors) for i in range(num_class)]
    gradients.append(inter_tensors)
    return input_tensor, gradients, sess

def integratedGrad(array, input_tensor, gradients: list, sess, n=10):
    """
    function: preparing left term values of equation (1) of the paper https://arxiv.org/pdf/1910.00387.pdf
    inputs: image array, graph paths: input_tensor and gradients
    outputs: value of the actual activations and gradients for the intermediate images
    """

    reference_value = np.zeros_like(array)
    step_size = (array - reference_value)/float(n)
    intermediate_values = [reference_value + j*step_size for j in range(n+1)] # n+1 inputs to feed the model
    feed_dict = {input_tensor: intermediate_values}
    run_output = sess.run(gradients, feed_dict = feed_dict) #len = 5
    activs = run_output[-1] # len = numLayer # activs[0].shape = (n,128,128,3)
    grads = run_output[:-1] # len = num_class
    return grads, activs



def get_convLayer(model):
    """ return convolutional layers of the input resnet model """

    convLayers = []
    for i in range(len(model.layers)):
        if "add" in model.layers[i].name:
            convLayers.append(model.layers[i+1])
    return convLayers


def getBlocks(convLayers: list):
    """ group layers in convLayers into layer of same channel number """

    convBlocks = []
    debut = 0
    current = convLayers[0].output.shape[-1]
    for i in range(1, len(convLayers)):
        fm_nb = convLayers[i].output.shape[-1]
        if current != fm_nb:
            end = i
            convBlocks.append(convLayers[debut:end])
            debut = end
            current = fm_nb
    end = len(convLayers)
    convBlocks.append(convLayers[debut:end])
    return convBlocks

def calculate_cond(model, x_train: list, train_pred: list, index: int, num_class: int = 10):
    """
    Function: calculate conductance of predictions made a model
    Inputs: the model and predictions it made
    Outputs: conductance values for each prediction
    """

    #get all conv layer in the network
    #for resnet, only the output of resBlocks are considered (skip connection + normal connection)
    convLayers = get_convLayer(model)
                
    #group conv layers by number of channel
    convBlocks = getBlocks(convLayers)
    
    allBlockCond = []
    for block in convBlocks:
        #input_tensor is the input tensor given to the model
        #outputs are gradients and activations associated to the output of block 
        input_tensor, gradients, sess = initialization(model, num_class, block, True)
        originIndex = {}
        conductances = {}

        #iterating over all wrong/correct prediction to calculate their conductance
        for i in tqdm(range(len(index))):
            array = x_train[index[i]]
            pred = train_pred[index[i]]
            #print(i, end = "\r")

            #calculate the gradient of the predicted index wrt to the activation output of block
            grads, activs = integratedGrad(array, input_tensor, [gradients[pred], gradients[-1]], sess)
            grads = np.array(grads)
            activs = np.array(activs)

            #calculate the conductance for neurons/outputs of block
            delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
            cond = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
            cond = np.mean(cond, axis = (0,2,3)) #averaging over axis 0 for purpose of squeezing

            #store conductance grouped by the class in which the image is predicted
            #originIndex store the original index of the image so its true label can be found
            if pred not in conductances.keys():
                originIndex[pred] = []
                conductances[pred] = []
                originIndex[pred].append(index[i])
                conductances[pred].append(cond)
            else:
                originIndex[pred].append(index[i])
                conductances[pred].append(cond)
                
        allBlockCond.append(conductances)

    #reformatting the conductances of all layers
    conductances = allBlockCond[0]
    for c in range(num_class):
        for i in range(len(conductances[c])):
            conductances[c][i] = list(conductances[c][i])
            for b in range(1, len(convBlocks)):
                conductances[c][i].extend(list(allBlockCond[b][c][i]))

    return conductances, originIndex 
  

def calculate_LCR(model, x_train: list, index: int, model_mutants: list, num_class: int = 10):
    """
    Function: calculate the label change rate of a model using its mutants
    Inputs: a model, its mutants and a training set to evaluate prediction stability
    Outputs: LCR corresponding to each prediction
    """ 
    
    #create a list that map the LCR to the conductance of the good training sample
    L = []
    for c in range(num_class):
        L.append(index[c])
    L = np.concatenate(L, axis = 0)
    
    #reload the orignal model each time else model would be the last mutant
    model.load_weights('cifar10resnet_weights.h5')
    pred = np.argmax(model.predict(x_train[L]), axis = 1)
    LCR = np.zeros(len(pred))
    
    #iterate over mutant to use them to predict x
    for n in range(len(model_mutants)):
        print("---- n="+str(n)+" ----", end = "\r")
        model.set_weights(model_mutants[n])
        mutate_pred = np.argmax(model.predict(x_train[L]), axis = 1)
        difference = mutate_pred != pred
        LCR[difference] += 1
    
    LCR = LCR/len(model_mutants)
    return LCR


def report(testD_res: list, testL: list, thresh: int, Print: bool):
    """ 
    Function: print several performance metrics related to input predictions and labels
    Inputs: prediction, labels and a decision threshold
    Outputs: accuracy and recall scores
    """

    acc = 0
    recallR = 0
    recallW = 0
    nbR = 0
    nbW = 0
    for i in range(len(testL)):
        if testL[i] == 0:
            nbW += 1
            if testD_res[i] < thresh:
                recallW += 1
                acc += 1
        if testL[i] == 1:
            nbR += 1
            if testD_res[i] > thresh:
                recallR += 1
                acc += 1
    acc = acc/len(testD_res)
    recallR, recallW = recallR/nbR, recallW/nbW
    recallM = (recallR + recallW)/2
    if Print == True:
        print("right recall", recallR) 
        print("wrong recall", recallW)  
        print("mean recall", recallM)
        print("accuracy", acc) 
  
    
def help_process_cond(cond_wrongs: dict, layer: int):
    """
    function: helper to the function process_cond, extract conductance of a specified layer.
    inputs: conductances and the layer.
    outputs: conductances of that layer.
    """
    finalWrongs = {}
    for c in sorted(cond_wrongs.keys()):
        cWrongs = []
        for i in range(len(cond_wrongs[c])):
            cWrongs.append(cond_wrongs[c][i][layer].reshape((1,-1)))
        cWrongs = np.concatenate(cWrongs)
        finalWrongs[c] = cWrongs
    return finalWrongs

def process_cond(layers: list, cond_wrongs: dict, concat: int = 0):
    """
    function: extract conductance of specific layers and organize them into tabular data
              where first columns are conductance of the last layers and last columns are conductance
              of the first layers.
    inputs: a list of layers with the first element being the last layer in the model.
    outpus: tabular data version of conductances.
    """
    finalWrongs = help_process_cond(cond_wrongs, layers[0])
    finalWrongs = list(finalWrongs.values())
    finalWrongs = np.concatenate(finalWrongs)

    # include other layers' conductance when concat > 0
    if concat > 0:
        for i in range(concat):
            concatWrongs = help_process_cond(cond_wrongs, layers[i+1])
            concatWrongs = list(concatWrongs.values())
            concatWrongs = np.concatenate(concatWrongs)
            finalWrongs = np.concatenate((finalWrongs, concatWrongs), axis = 1)
    return finalWrongs
            

def error_detector(input_shape: tuple, loss_function: str = 'binary_crossentropy', lr: float = 0.001):
    """ create a binary classifier neural network with specified parameters """

    net = Sequential()    
    net.add(Dense(200, input_shape = (input_shape,), activation = 'relu'))
    net.add(Dropout(0.5))
    net.add(Dense(200, activation = 'relu'))
    net.add(Dropout(0.5))
    net.add(Dense(1, activation = 'sigmoid'))
    opt = keras.optimizers.Adam(lr=lr)
    net.compile(loss=loss_function,optimizer=opt,metrics=None)
    return net


def find_opti_thresh(testL: list, pred: list, plot: bool = True):
    """
    function: using roc curve created from label and prediction, find the threshold that
              minimize the difference of recall score for both classes
    inputs: label testL, predictions pred
    outputs: an optimal threhsold 
    """
    fpr, tpr, thresholds = roc_curve(testL, pred)
    roc_auc = auc(fpr,tpr)
    maxi = 0
    max_ind = 0
    for i in range(len(fpr)):
        s = 1 - fpr[i] + tpr[i]
        if s > maxi:
            maxi = s
            max_ind = i
    if plot == True:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return thresholds, max_ind, roc_auc
    


from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return 

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,5)),str(round(roc_val,5))),end=100*' '+'\n')
        return roc_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return






