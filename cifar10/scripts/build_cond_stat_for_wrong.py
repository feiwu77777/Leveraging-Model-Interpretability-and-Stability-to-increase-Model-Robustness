# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:22:37 2019

@author: fwursd
"""

train_pred = np.argmax(model.predict(x_train), axis = 1)
train_label = np.argmax(y_train, axis = 1)

w = train_label != train_pred
w_ind = np.arange(len(w))[w]

allClassWrongCond = []
input_tensor, outputs, sess = layerInit(model, num_class, convLayers, True)
for c in range(num_class):
    class_cond = []
    print(c)
    for i in range(len(w_ind)):
        print(i, end = '\r')
        array = x_train[w_ind[i]]
        pred = np.argmax(model.predict(np.expand_dims(array,0)))
        if pred == c:
            contribs = layerOneCond(array, pred, convLayers, num_class, input_tensor, outputs,sess)
            class_cond.append(contribs)
    allClassWrongCond.append(class_cond)
    
pickle.dump(allClassWrongCond, open('allClassWrongCondCifar10.p', 'wb'))

allClassWrongCond = pickle.load(open('allClassWrongCondCifar10.p', 'rb'))
allWrongTop = []
for Class in allClassWrongCond:
    allWrongTop.append(layerTopCond(Class, 2, convLayers))
    
ind = 9
allWrongTop[ind]['conv8']

allWrongTopNBN[ind]