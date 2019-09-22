# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:20:14 2019

@author: fwursd
"""

input_tensor, outputs, sess = layerInit(model, num_class, convLayers, True)
cond = layerOneCond(x_test[0], 0, convLayers, num_class, input_tensor, outputs,sess, sort = False, limit = None)


allTopTrain = pickle.load(open('train_cond.p','rb'))
allTop = []
for Class in allTopTrain:
    allTop.append(layerTopCond(Class, 2, convLayers))
rank = [4,4,4,4,4,4,4,4,4]
topFMs, topMeans = elisor(rank, convLayers, allTop, num_class)

wrongs = pickle.load(open("wrong_train_cond.p", "rb"))

def process(convLayers,topFMs, topMeans, img, class_ind):
    finalWrongs = [[] for i in range(len(convLayers))]
    label = np.zeros(10)
    label[class_ind] = 1
    for l in range(len(convLayers)):
        normalized = img[l][topFMs[l,class_ind][:]]/topMeans[l,class_ind][:]
        if l == len(convLayers)-1:
            normalized = np.concatenate((label,normalized))
        finalWrongs[l].append(normalized)
    finalWrongs = np.concatenate([finalWrongs[i] for i in reversed(range(len(convLayers)))], axis = 1)
    return finalWrongs

finalWrong = process(convLayers,topFMs, topMeans, cond, 0)