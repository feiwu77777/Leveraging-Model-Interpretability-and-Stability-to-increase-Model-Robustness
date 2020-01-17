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
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc



def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

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


def resnet_v1(input_shape: tuple, depth: int, num_classes: int = 10) -> Model:
    """ Construct a resnet with specified parameters """

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
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def topCond(conductances, rank):
    nb = len(conductances)
    convDict = {}
    for img in conductances:
        layer = img
        for featureMap in layer[:rank]:
            if featureMap[0] not in convDict.keys():
                convDict[featureMap[0]] = (1,[featureMap[1]],0,0)
            else:
                convDict[featureMap[0]] = (convDict[featureMap[0]][0]+1, convDict[featureMap[0]][1],0,0)
                convDict[featureMap[0]][1].append(featureMap[1])
    for key in convDict.keys():
        percentage = np.round(convDict[key][0]*100/nb,decimals=2)
        mean = np.mean(convDict[key][1])
        std = np.std(convDict[key][1])
        convDict[key] = (percentage,convDict[key][1],mean,std)
        

    dicti = {}
    L = []
    for Map in convDict.items():
        L.append((Map[0],Map[1][0],Map[1][1],Map[1][2],Map[1][3]))
    L = sorted(L, key = lambda x: x[1], reverse = True)
    for l in L:
        dicti[l[0]] = (str(l[1])+'%',l[3],l[4])
    return dicti

def layerTopCond(conductances, rank, convLayers):
    nb = len(conductances)
    FMCond = {}
    for i in range(len(convLayers)):
        convDict = {}
        for img in conductances:
            layer = img[i]
            for featureMap in layer[:rank]:
                if featureMap[0] not in convDict.keys():
                    convDict[featureMap[0]] = (1,[featureMap[1]],0,0)
                else:
                    convDict[featureMap[0]] = (convDict[featureMap[0]][0]+1, convDict[featureMap[0]][1],0,0)
                    convDict[featureMap[0]][1].append(featureMap[1])
        for key in convDict.keys():
            percentage = np.round(convDict[key][0]*100/nb,decimals=2)
            mean = np.mean(convDict[key][1])
            std = np.std(convDict[key][1])
            convDict[key] = (percentage,convDict[key][1],mean,std)
        FMCond["conv"+str(i)] = convDict
        
    FMC = {}
    for i in range(len(convLayers)):
        dicti = {}
        L = []
        for Map in FMCond["conv"+str(i)].items():
            L.append((Map[0],Map[1][0],Map[1][1],Map[1][2],Map[1][3]))
        L = sorted(L, key = lambda x: x[1], reverse = True)
        for l in L:
            dicti[l[0]] = (str(l[1])+'%',l[3],l[4])
        FMC["conv"+str(i)] = dicti

    return FMC


def extractTop(rank,convLayers, allTop, num_class):
    topFMs = []
    topMeans = []
    for l in range(len(convLayers)):
        topFM = []
        topMean = []
        for k in range(num_class):
            topFM.append(np.array(list(allTop[k]["conv"+str(l)].keys())[:rank[l]]))
            fmMeans = list(allTop[k]["conv"+str(l)].values())[:rank[l]]
            fmMeans = [mean[1] for mean in fmMeans]
            topMean.append(np.array(fmMeans))
        topFMs.append(topFM)
        topMeans.append(topMean)
    topFMs = np.array(topFMs)
    topMeans = np.array(topMeans)
    return topFMs, topMeans

def process_cond(convLayers,topFMs, topMeans, wrongs, num_class):
    finalWrongs = [[] for i in range(len(convLayers))]
    for k in range(num_class):
        label = np.zeros(num_class)
        label[k] = 1
        for img in wrongs[k]:
            for l in range(len(convLayers)):
                normalized = img[l][topFMs[l,k][:]]/topMeans[l,k][:]
                if l == len(convLayers)-1:
                    normalized = np.concatenate((label,normalized))
                finalWrongs[l].append(normalized)
    finalWrongs = np.concatenate([finalWrongs[i] for i in reversed(range(len(convLayers)))], axis = 1)
    return finalWrongs


def calculate_LCR(model, x_train, index, model_mutants, num_class = 10):
    
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

def get_convLayer(model):
    convLayers = []
    for i in range(len(model.layers)):
        if "add" in model.layers[i].name:
            convLayers.append(model.layers[i+1])
    return convLayers

def calculate_cond(model, x_train, train_pred, index, num_class = 10):

    #get all conv layer in the network
    #for resnet, only the output of resBlocks are considered (skip connection + normal connection)
    convLayers = get_convLayer(model)
                
    #group conv layers by number of channel
    convBlocks = getBlocks(convLayers)
    
    allBlockCond = []
    for block in convBlocks:
        #input_tensor is the input tensor given to the model
        #outputs are gradients and activations associated to the output of block 
        input_tensor, outputs, sess = layerInit(model, num_class, block, True)
        originIndex = {}
        conductances = {}

        #iterating over all wrong/correct prediction to calculate their conductance
        for i in range(len(index)):
            array = x_train[index[i]]
            pred = train_pred[index[i]]
            print(i, end = "\r")

            #calculate the gradient of the predicted index wrt to the activation output of block
            grads, activs = integratedGrad(array, input_tensor, [outputs[pred],outputs[-1]], sess)
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
  
def report_acc(testD_res, testL, thresh, Print):
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
        print("accuracy", acc, "mean recall", recallM, "right recall", recallR, "wrong recall", recallW)
    return acc, recallM, recallR, recallW
    
    
def layerInit(model, num_class,  inter_layer, output_activ):
    sess = K.get_session()
    input_tensor = model.layers[0].input
    inter_tensors = [conv.output for conv in inter_layer]
    if output_activ == True:
        out = model.layers[-2].output
        weights, bias = model.layers[-1].weights
        output_tensor = tf.add(tf.matmul(out,weights),bias)
    else:
        output_tensor = model.layers[-2].output
    outputs = [tf.gradients(output_tensor[:,i],inter_tensors) for i in range(num_class)]
    outputs.append(inter_tensors)
    return input_tensor, outputs, sess

def integratedGrad(array, input_tensor, outputs, sess, n=10):
    reference_value = np.zeros_like(array)
    step_size = (array - reference_value)/float(n)
    intermediate_values = [reference_value + j*step_size for j in range(n+1)] # n+1 inputs to feed the model
    feed_dict = {input_tensor: intermediate_values}
    run_output = sess.run(outputs, feed_dict = feed_dict) #len = 5
    activs = run_output[-1] # len = numLayer # activs[0].shape = (n,128,128,3)
    grads = run_output[:-1] # len = num_class
    return grads, activs


def oneCond(array, class_ind, sort = True, limit = None):
    grads, activs = integratedGrad(array, input_tensor, outputs, sess)
    grads = np.array(grads)
    activs = np.array(activs)
    delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:] #shape = (n-1,128,128,3)
    contribs = np.sum(grads[:,:,1:,:,:]*delta_activs, axis = 2)
    contribs = np.mean(contribs, axis = (2,3))
    target_contrib = contribs[class_ind,:,:] # list of length num channel of layer j
    meanCon = np.mean(contribs,axis=0) #shape = (64,)  # mean over the 4 label conductances
    final_contrib = list(target_contrib - meanCon) # len = channel number
    if sort == True:
        for l in range(len(final_contrib)):
            final_contrib[l] = list(final_contrib[l])
            for p in range(len(final_contrib[l])):
                final_contrib[l][p] = (p,final_contrib[l][p])
            final_contrib[l] = sorted(final_contrib[l], key = lambda x: x[1], reverse = True) #order by fm conductances
    if limit == None:
        return final_contrib
    else:
        return final_contrib

def layerOneCond(array, class_ind, convLayers, num_class, input_tensor, outputs,sess, sort = True, limit = None):
    grads, activs = integratedGrad(array, input_tensor, outputs, sess)
    layer_contrib = []
    for i in range(len(convLayers)):
        delta_activs = activs[i][1:,:,:,:] - activs[i][:-1,:,:,:] #shape = (n-1,128,128,3)
        contribs = []
        for j in range(num_class):
            contrib = np.sum(grads[j][i][1:,:,:,:]*delta_activs, axis = 0)
            contrib = np.mean(contrib, axis = (0,1))
            contribs.append(contrib)
        target_contrib = contribs[class_ind] # list of length num channel of layer j
        meanCon = np.mean(contribs,axis=0) #shape = (64,)  # mean over the 4 label conductances
        final_contrib = target_contrib - meanCon # len = channel number
        if sort == True:
            final_contrib = list(final_contrib)
            for p in range(len(final_contrib)):
                final_contrib[p] = (p,final_contrib[p])
            final_contrib = sorted(final_contrib, key = lambda x: x[1], reverse = True) #order by fm conductances
        layer_contrib.append(final_contrib)
    return layer_contrib


def new_conf(cm, num_class, thresh, true_label, allLayer, verif, finalWrongs, finalRights):
    new_confusion = np.zeros_like(cm)
    for class_ind in range(num_class):
        if class_ind == 0:
            wrongs = finalWrongs[:len(true_label[class_ind])]
            rights = finalRights[:len(allLayer[class_ind])]
        else:
            wrongs = finalWrongs[len(true_label[class_ind-1]):len(true_label[class_ind-1]) + len(true_label[class_ind])]
            rights = finalRights[len(allLayer[class_ind-1]):len(allLayer[class_ind-1])+len(allLayer[class_ind])]
        resW = verif.predict(wrongs)
        resR = verif.predict(rights)
        dicti = {}
        for i in range(len(resW)):
            if resW[i] <= thresh:
                if true_label[class_ind][i] not in dicti.keys():
                    dicti[true_label[class_ind][i]] = 1
                else:
                    dicti[true_label[class_ind][i]] += 1
        for i in dicti.keys():
            new_confusion[i, class_ind] = cm[i, class_ind] - dicti[i]

        nb = 0
        for i in range(len(resR)):
            if resR[i] > thresh:
                nb += 1
        new_confusion[class_ind, class_ind] = nb
    return new_confusion

def plot_new_confusion(old_cm, new_cm, classes, figsize):
    
    cmap=plt.cm.Blues
    diff_cm = old_cm - new_cm
    tot_good = np.sum([old_cm[i,i] for i in range(old_cm.shape[0])])
    tot_wrong = np.sum(old_cm) - tot_good
    summ = np.zeros((diff_cm.shape[1],2), dtype = 'int64')
    for i in range(diff_cm.shape[1]):
        summ[i,0] = diff_cm[i,i]
        summ[i,1] = np.sum(diff_cm[i,:]) - diff_cm[i,i]
    print(str(np.sum(summ, axis = 0)[0])+' good predictions are contested. Total good predictions: '+str(tot_good))
    print(str(np.sum(summ, axis = 0)[1])+' wrong predictions are contested. Total wrong predictions: '+str(tot_wrong))
    all_cm = [old_cm, new_cm, diff_cm]
    
    fig, axes = plt.subplots(1,3, figsize = figsize)
    axes[0].imshow(old_cm, interpolation='nearest', cmap=cmap)
    axes[1].imshow(new_cm, interpolation='nearest', cmap=cmap)
    axes[2].imshow(diff_cm, interpolation='nearest', cmap=cmap)
    #axes[3].imshow(summ, interpolation = 'nearest', cmap=cmap)
    axes[0].set_title("original confusion matrix")
    axes[0].set_xlabel('predicted')
    axes[1].set_xlabel('predicted')
    axes[2].set_xlabel('predicted')
    axes[0].set_ylabel('actual')
    axes[1].set_ylabel('actual')
    axes[2].set_ylabel('actual')
    axes[1].set_title("new confusion matrix")
    axes[2].set_title("contested images")

    fmt = 'd'
    for i, j in itertools.product(range(old_cm.shape[0]), range(old_cm.shape[1])):
        for k in range(3):
            thresh = all_cm[k].max() / 2.
            axes[k].text(j, i, format(all_cm[k][i, j], fmt),
                     horizontalalignment="center",
                     color="white" if all_cm[k][i, j] > thresh else "black")
#     thresh = summ.max() / 2.
#     for i, j in itertools.product(range(summ.shape[0]), range(summ.shape[1])):
#         axes[3].text(j, i, format(summ[i, j], fmt),
#                      horizontalalignment="center",
#                      color="white" if summ[i, j] > thresh else "black")
            
            
def find_opti_thresh(testL, pred, plot = True):
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


def augment(finalWrongs, LCRW, more_finalRights, more_LCRR, num_class):
    more_finalWrongs = np.zeros(more_finalRights.shape)
    more_LCRW = np.zeros(len(more_LCRR))
    for i in range(len(more_finalRights)):
        rnd_ind = np.random.randint(0,len(finalWrongs))
        row = finalWrongs[rnd_ind]
        rowLCR = LCRW[rnd_ind]
        pertLCR = np.random.choice([-0.01,0.01])
        more_LCRW[i] = rowLCR + pertLCR
        more_finalWrongs[i,:num_class] = row[:num_class]
        for j in range(num_class, len(row)):
            pert = np.random.choice([-0.01,0.01])
            more_finalWrongs[i,j] = row[j] + pert
    return more_finalWrongs, more_LCRW


def condHeat(array, model, num_class, convLayers, fm = None):
    input_tensor, outputs, sess = layerInit(model, num_class, convLayers[-1:], True)
    grads,activs = integratedGrad(array, input_tensor, outputs, sess)
    grads, activs = np.array(grads), np.array(activs)
    delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:]
    contribs = np.sum(grads[:,:,1:,:,:]*delta_activs, axis = 2)
    contribs = np.squeeze(contribs, axis = 1)
    mean = np.mean(contribs, axis=0)
    contribs = contribs - mean
    contribs = list(contribs)
    if fm == None:
        contribs = [np.mean(contrib,axis=2)*100 for contrib in contribs]
    else:
        contribs = [contrib[:,:,fm] for contrib in contribs] 
    return contribs

def condViz(heatmap, array, num_class, index_to_label, figsize = (20,12), xAxis = 5):

    yAxis = np.int(np.ceil((num_class+2)/xAxis))
    fig, axes = plt.subplots(yAxis,xAxis, figsize = figsize)
    k = 0
    for i in range(yAxis):
        for j in range(xAxis):
            if (i==0)&(j==0):
                axes[i,j].imshow(image.array_to_img(array))
                axes[i,j].grid(True)
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
            else:
                try:
                    sns.heatmap(heatmap[k],center = 0, annot = True,yticklabels = False, xticklabels = False, ax = axes[i,j])
                    axes[i,j].set_title(index_to_label[k], fontsize = 20)
                    k += 1
                except:
                    axes[i,j].imshow(image.array_to_img(array))
                    axes[i,j].grid(True)
                    axes[i,j].set_xticks([])
                    axes[i,j].set_yticks([])
                    break
                
                
def load_binary(weight = None):
    binary = Sequential()    
    binary.add(Dense(100, activation = 'relu', input_shape = (46,)))
    binary.add(BatchNormalization())
    binary.add(Dense(500, activation = 'relu'))
    binary.add(BatchNormalization())
    binary.add(Dropout(0.5))
    binary.add(Dense(500, activation = 'relu'))
    binary.add(BatchNormalization())
    binary.add(Dropout(0.5))
    binary.add(Dense(100, activation = 'relu'))
    binary.add(BatchNormalization())
    binary.add(Dropout(0.5))
    binary.add(Dense(1, activation = 'sigmoid'))
    if weight == None:
        return binary
    else:
        binary.load_weights(weight)
        return binary
    

def preprocess_PCA(wrongs, convLayers):
    finalWrongs = []
    for l in range(len(convLayers)):
        lWrongs = {}
        for c in sorted(wrongs.keys()):
            cWrongs = []
            for i in range(len(wrongs[c])):
                cWrongs.append(wrongs[c][i][l].reshape((1,-1)))
            cWrongs = np.concatenate(cWrongs)
            lWrongs[c] = cWrongs
        finalWrongs.append(lWrongs)
    return finalWrongs

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

def oneL_preprocess_PCA(wrongs, layer, num_class):
    finalWrongs = {}
    for c in sorted(wrongs.keys()):
        cWrongs = []
        for i in range(len(wrongs[c])):
            cWrongs.append(wrongs[c][i][layer].reshape((1,-1)))
        cWrongs = np.concatenate(cWrongs)
        finalWrongs[c] = cWrongs
    return finalWrongs


def verif_data(rg, rank, num_class, brut_data, brut_labels):
#     brut_data = brut_data[:,:num_class+np.sum([rank[len(rank) - i - 1] for i in range(rg)])]
    print("brut data shape is ", brut_data.shape)
    rnd = np.arange(len(brut_labels))
    np.random.seed(0)
    np.random.shuffle(rnd)
    finalData = brut_data[rnd]
    finalLabels = brut_labels[rnd]
    split = 0.2
    split_ind = int(len(finalData)-split*len(finalData))
    trainD = finalData[:split_ind]
    trainL = finalLabels[:split_ind]
    testD = finalData[split_ind:]
    testL = finalLabels[split_ind:]
    return trainD, trainL, testD, testL, rnd


def opti_nestim(n_estimators, trainD, trainL, testD, testL):
    train_results = []
    test_results = []
    count = 0
    for estimator in n_estimators:
        print(count, end = '\r')
        count += 1
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(trainD, trainL)
        train_pred = rf.predict(trainD)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(trainL, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(testD)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(testL, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
    line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()
    
def opti_depth(max_depths, trainD, trainL, testD, testL):
    train_results = []
    test_results = []
    count = 0
    for max_depth in max_depths:
        print(count, end = '\r')
        count += 1
        rf = RandomForestClassifier(n_estimators = 75, max_depth=max_depth, n_jobs=-1)
        rf.fit(trainD, trainL)
        train_pred = rf.predict(trainD)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(trainL, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(testD)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(testL, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()
    
def getBlocks(convLayers):
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


def create_verif(input_shape, loss_function = 'binary_crossentropy', lr = 0.001):
    verif = Sequential()    
    verif.add(Dense(200, input_shape = (input_shape,), activation = 'relu'))
    verif.add(Dropout(0.5))
    verif.add(Dense(200, activation = 'relu'))
    verif.add(Dropout(0.5))
    verif.add(Dense(1, activation = 'sigmoid'))
    opt = keras.optimizers.Adam(lr=lr)
    verif.compile(loss=loss_function,optimizer=opt,metrics=['accuracy'])
    return verif


def oneL_fit_PCA(finalWrongs, finalRights, num_class, n_components, scale):
    PCAs = {}
    for c in range(num_class):
        if c in finalWrongs.keys():
            data = np.concatenate((finalWrongs[c], finalRights[c]))
        else:
            data = finalRights[c]
            
        pca = PCA(n_components = n_components)
        pca.fit(data)
        PCAs[c] = pca
        
        if c in finalWrongs.keys():
            class_vecW = np.zeros((finalWrongs[c].shape[0],num_class))
            class_vecW[:,c] += 1
            finalWrongs[c] = pca.transform(finalWrongs[c])*scale
            finalWrongs[c] = np.concatenate((class_vecW, finalWrongs[c]), axis = 1)
        
        class_vecR = np.zeros((finalRights[c].shape[0],num_class))
        class_vecR[:,c] += 1
        finalRights[c] = pca.transform(finalRights[c])*scale
        finalRights[c] = np.concatenate((class_vecR, finalRights[c]), axis = 1)


    finalWrongs = list(finalWrongs.values()) 
    finalRights = list(finalRights.values())
    finalWrongs = np.concatenate(finalWrongs)
    finalRights = np.concatenate(finalRights)
    return PCAs, finalWrongs, finalRights

def oneL_fitTest_PCA(test_cond, layer, num_class, PCAs, scale):
    finalTest = oneL_preprocess_PCA(test_cond, layer, num_class)
    FinalTest = {}
    for c in sorted(PCAs.keys()):
        pca = PCAs[c]
        finalTest[c] = pca.transform(finalTest[c])*scale
        class_vec = np.zeros((finalTest[c].shape[0], num_class))
        class_vec[:,c] += 1

        finalTest[c] = np.concatenate((class_vec, finalTest[c]), axis = 1)
        FinalTest[c] = finalTest[c]
    
    FinalTest = list(FinalTest.values())
    FinalTest = np.concatenate(FinalTest)
    return FinalTest

def create_weighted_binary_crossentropy(zero_weight, one_weight):
    
    def weighted_binary_crossentropy(y_true, y_pred):
        
        b_ce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def process_model_params(model):
    params_num = 0
    allWeights = []
            
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name:
            weights = model.layers[i].get_weights()[0]
            params = weights.shape[-1]
            allWeights.append((i,np.arange(params_num,params_num+params)))
            params_num += params
    return params_num, allWeights

def create_mutations(mode, model, params_num, allWeights, 
                     origin_weight, gamma, test_acc, x_test, y_label, limit = 100):

    #mode is either Gaussian Fuzz or Neuron Activation Inverse. Both are different kind of operation to modify weights.
    #gamma is the percentage of weights to be modified

    mutations = []
    count = 0
    while len(mutations) != limit:
        print(count, end = "\r")
        count += 1
        model.load_weights(origin_weight)
        rnd = np.arange(params_num)
        np.random.shuffle(rnd)
        rnd = rnd[:int(params_num*gamma)]
        rnd = sorted(rnd)
        for i in range(len(allWeights)):
            for num in rnd:
                if num in allWeights[i][1]:
                    index = np.argwhere(allWeights[i][1] == num).item()
                    w = model.layers[allWeights[i][0]].get_weights()[0]
                    if mode == 'GF':
                        avg_w = np.mean(w, axis = -1)
                        std_w = np.std(w, axis = -1)
                        w[:,:,:,index] = np.random.normal(avg_w, std_w)
                    if mode == 'NAI':
                        w[:,:,:,index] = -1*w[:,:,:,index]
                    model.layers[allWeights[i][0]].set_weights([w,b])

        test_pred = np.argmax(model.predict(x_test), axis = 1)
        test_acc = np.mean(test_pred == test_label)
        print(acc)
        if acc > 0.9*test_acc:
            mutations.append(model.get_weights())
    return mutations


def calc_LCR(model, x, mutations):
    model.load_weights('cifar10resnet_weights.h5')
    pred = np.argmax(model.predict(x), axis = 1)
    LCR = np.zeros(len(x))
    for n in range(len(mutations)):
        print("---- n="+str(n)+" ----", end = "\r")
        model.set_weights(mutations[n])
        mutated_pred = np.argmax(model.predict(x), axis = 1)
        LC = mutated_pred != pred
        LCR[LC] += 1
    LCR = LCR/len(mutations)
    return LCR


