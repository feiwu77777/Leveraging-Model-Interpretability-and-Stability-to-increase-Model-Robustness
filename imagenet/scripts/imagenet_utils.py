
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
import keras
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import seaborn as sns
import pickle


def create_net(weights_path, num_class):
    net = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (299,299,3))
    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_class)(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs = net.input, outputs = outputs)
    model.load_weights(weights_path)
    return model

def load(class_ind,img_ind,path):
    classes = sorted(os.listdir(path))
    class_img = sorted(os.listdir(path+classes[class_ind]))
    img = image.load_img(path+classes[class_ind]+'/'+class_img[img_ind], target_size = (299,299))
    return img

def load_array(class_ind, img_ind, path):
    img = load(class_ind, img_ind, path)
    array = preprocess_input(image.img_to_array(img))
    return array

def init(inter_layer, model, num_class):
    sess = K.get_session()
    input_tensor = model.layers[0].input
    inter_tensors = [conv.output for conv in inter_layer]
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
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    return thresholds, max_ind, roc_auc

def report_acc(testD_res, testL, thresh, Print):
    acc = 0
    recallR = 0
    recallW = 0
    nbR = 0
    nbW = 0
    for i in range(len(testL)):
        if testL[i] == 1:
            nbW += 1
            if testD_res[i] > thresh:
                recallW += 1
                acc += 1
        if testL[i] == 0:
            nbR += 1
            if testD_res[i] < thresh:
                recallR += 1
                acc += 1
    acc = acc/len(testD_res)
    recallR, recallW = recallR/nbR, recallW/nbW
    recallM = (recallR + recallW)/2
    if Print == True:
        print("accuracy", acc, "mean recall", recallM, "right recall", recallR, "wrong recall", recallW)
    return acc, recallM, recallR, recallW

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



def preprocess_cond(wrongs, convLayers):
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


def oneG_extractTop(allClassCond2, num_class, convLayers, rank = 4):
    
    #catching the mean of every fm across all positive img
    allFmMean = []
    for c in range(num_class):
        classFmMean = []
        classCond = allClassCond2[c]
        for l in range(len(convLayers)):
            grouping = np.array([classCond[i][l] for i in range(len(classCond))])
            mean = np.mean(grouping, axis = 0)
            classFmMean.append(mean)
        allFmMean.append(classFmMean)
        
    #calc the mean of a fm across all classes
    allClassMeans = []
    for l in range(len(convLayers)):
        to = np.array([allFmMean[c][l] for c in range(num_class)])
        to = np.mean(to, axis = 0)
        allClassMeans.append(to)
        
    #substracting the average cond across classes of a fm to this fm
    #then rank all fm of all layer and classes
    for c in range(num_class):
        for l in range(len(convLayers)):
            allFmMean[c][l] -= allClassMeans[l]
            allFmMean[c][l] = list(allFmMean[c][l])
            allFmMean[c][l] = [(i, allFmMean[c][l][i]) for i in range(len(allFmMean[c][l]))]
            allFmMean[c][l] = sorted(allFmMean[c][l], key = lambda x: x[1], reverse = True)
    #check top fm for all classes
    topFMs = []
    topMeans = []
    for l in range(len(convLayers)):
        classTop = []
        classMean = []
        for c in range(num_class):
            cac = allFmMean[c][l][:rank]
            classTop.append([ca[0] for ca in cac])
            classMean.append([ca[1] for ca in cac])
        topFMs.append(classTop)
        topMeans.append(classMean)
    topFMs = np.array(topFMs)
    topMeans = np.array(topMeans)
    
    return topFMs, topMeans, allClassMeans

def oneL_preprocess_cond(wrongs, layer, num_class):
    finalWrongs = {}
    for c in sorted(wrongs.keys()):
        cWrongs = []
        for i in range(len(wrongs[c])):
            cWrongs.append(wrongs[c][i][layer].reshape((1,-1)))
        cWrongs = np.concatenate(cWrongs)
        finalWrongs[c] = cWrongs
    return finalWrongs

def process_data(layers, wrongs2, concat = 0):
    finalWrongs = oneL_preprocess_cond(wrongs2, layers[0], num_class)
    finalWrongs = list(finalWrongs.values())
    finalWrongs = np.concatenate(finalWrongs)
    if concat > 0:
        for i in range(concat):
            concatWrongs = oneL_preprocess_PCA(wrongs2, layers[i+1], num_class)
            concatWrongs = list(concatWrongs.values())
            concatWrongs = np.concatenate(concatWrongs)
            finalWrongs = np.concatenate((finalWrongs, concatWrongs), axis = 1)
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

def plot_accs(NCs, gen_accs, r_accs, w_accs, m_accs, layer):
    plt.plot(NCs, gen_accs)
    plt.plot(NCs, r_accs)
    plt.plot(NCs, w_accs)
    plt.plot(NCs, m_accs)
    plt.title('layer '+str(layer)+' optimum PCAs')
    plt.ylabel('test accuracy')
    plt.xlabel('n_components')
    plt.legend(['y = gen_accs', 'y = r_accs', 'y = w_accs', 'y = m_accs'], loc='lower right')
    


def weighted_reportAcc(Wval_res, valL, Print):
    acc = np.sum(Wval_res == valL)/len(valL)
    nbR = 0
    nbW = 0
    recallR = 0
    recallW = 0
    for i in range(len(Wval_res)):
        if valL[i] == 1:
            nbR += 1
            if Wval_res[i] == 1:
                recallR += 1
        if valL[i] == 0:
            nbW += 1
            if Wval_res[i] == 0:
                recallW += 1
    recallR, recallW = recallR/nbR, recallW/nbW
    recallM = (recallR+recallW)/2
    if Print:
        print('accuracy', acc, 'mean recall', recallM, 'right recall', recallR, 'wrong recall', recallW)
    return acc, recallM, recallR, recallW


def plot_confusion_matrix(y_true, y_pred, classes_name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes_name
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]       

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def create_verif(input_shape, loss_function = 'binary_crossentropy'):
    verif = Sequential()    
    verif.add(Dense(200, input_shape = (input_shape,), activation = 'relu'))
    verif.add(Dropout(0.5))
    verif.add(Dense(200, activation = 'relu'))
    verif.add(Dropout(0.5))
    verif.add(Dense(1, activation = 'sigmoid'))
    opt = keras.optimizers.Adam(lr=0.001)
    verif.compile(loss=loss_function,optimizer=opt,metrics=None)
    return verif

def pred_ind(model, origin_ind, path, bs = 100):
    pred = []
    reset = 0
    for c in sorted(origin_ind.keys()):
        print(c, end = '\r')
        for i in range(len(origin_ind[c])):
            array = load_array(origin_ind[c][i][0], origin_ind[c][i][1], path)[None]
            if reset == 0:
                data = array
                reset = 1
            else:
                data = np.concatenate((data, array), axis = 0)
            if len(data) == bs:
                reset = 0
                bs_pred = np.argmax(model.predict(data), axis = 1)
                pred.append(bs_pred)
    if len(data) != bs:
        bs_pred = np.argmax(model.predict(data), axis = 1)
        pred.append(bs_pred)

    pred = np.concatenate(pred)
    return pred

def predict(model, path, bs = 100):
    classes = sorted(os.listdir(path))
    num_class = len(classes)
    pred = {c:[] for c in range(num_class)}
    conc = 0
    for c in range(num_class):
        print(c)
        for i in range(len(os.listdir(path+classes[c]))):
            print(i, end = '\r')
            array = load_array(c, i, path)[None]
            if conc == 0:
                data = array
                conc = 1
            else:
                data = np.concatenate((data, array), axis = 0)
            if len(data) == bs:
                conc = 0
                bs_pred = np.argmax(model.predict(data), axis = 1)
                pred[c].extend(list(bs_pred))
        if len(data) != bs:
            conc = 0
            bs_pred = np.argmax(model.predict(data), axis = 1)
            pred[c].extend(list(bs_pred))
    return pred

def calc_LCR(model, origin_ind, path, mutations):
    origin_pred = np.concatenate([[c for i in range(len(origin_ind[c]))] for c in sorted(origin_ind.keys())])
    LCR = np.zeros(len(origin_pred))
    for n in range(len(mutations)):
        print("---- n="+str(n)+" ----")
        model.set_weights(mutations[n])
        mutated_pred = pred_ind(model, origin_ind, path)
        LC = mutated_pred != origin_pred
        LCR[LC] += 1
    LCR = LCR/len(mutations)
    return LCR

def create_weighted_binary_crossentropy(zero_weight, one_weight):
    
    def weighted_binary_crossentropy(y_true, y_pred):
        
        b_ce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def create_mutations(mode, model, params_num, allWeights, origin_weight, gamma, val_acc, val_gen, limit = 100):       
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
                    model.layers[allWeights[i][0]].set_weights([w])
        acc = model.evaluate_generator(val_gen, steps = val_gen.n // val_gen.batch_size, verbose = 1)[1]
        print(acc)
        if acc > 0.9*val_acc:
            mutations.append(model.get_weights())
    return mutations


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


import itertools
def plot_new_confusion(k, old_cm, new_cm, figsize):
    
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
    
    fig, axes = plt.subplots(1,1, figsize = figsize)
    axes.imshow(old_cm, interpolation='nearest')
    axes.set_title("original confusion matrix")
    axes.set_xlabel('predicted')
    axes.set_ylabel('actual')
#     axes[1].imshow(new_cm, interpolation='nearest', cmap=cmap)
#     axes[1].set_xlabel('predicted')
#     axes[1].set_ylabel('actual')
#     axes[1].set_title("new confusion matrix")
#     axes[2].imshow(diff_cm, interpolation='nearest', cmap=cmap)
#     axes[2].set_xlabel('predicted')
#     axes[2].set_ylabel('actual')
#     axes[2].set_title("contested images")
    #axes[3].imshow(summ, interpolation = 'nearest', cmap=cmap)



    fmt = 'd'
    for i, j in itertools.product(range(old_cm.shape[0]), range(old_cm.shape[1])):
        #thresh = all_cm[k].max() / 2.
        axes.text(j, i, format(all_cm[k][i, j], fmt),
                 horizontalalignment="center",
                 color= "red" if (i != j)&(all_cm[k][i,j] != 0) else 'gray')
        
        
def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predCheck(model, layer, convLayers, class_ind, img_ind, path, topFMs, 
              index_to_label, train_pred, train_proba, allClassMeans, v = 0,
              figsize, plot = True, rank = True):
    
    array = load_array(class_ind, img_ind, path)
    pred = int(train_pred[class_ind, img_ind])
    prob = train_proba[class_ind, img_ind]
    
    sess = K.get_session()
    input_tensor = model.layers[0].input
    convOut = convLayers[layer]
    inter_tensors = [convOut.output]
    output_tensor = model.layers[-2].output
    outputs = [tf.gradients(output_tensor[:,pred],inter_tensors), inter_tensors]
    
    grads, activs = integratedGrad(array, input_tensor, outputs, sess)
    grads, activs = np.array(grads), np.array(activs)
    delta_activs = activs[:,1:,:,:,:] - activs[:,:-1,:,:,:] #shape = (n-1,128,128,3)
    contribs = np.sum(grads[:,:,1:,:,:,:]*delta_activs, axis = 2)
    contribs = np.squeeze(contribs, axis = (0,1))
    
    topFMs_pred = topFMs[layer,pred]
    contribs_pred = [contribs[:,:,fm] for fm in topFMs_pred]
    
    if plot:
        print("predicted as \'"+index_to_label[pred]+"\' with proba "+str(prob))
        print("is actually \'"+index_to_label[class_ind]+'\'')
    
        fig, axes = plt.subplots(1,len(topFMs_pred)+1, figsize = figsize)
        axes[0].imshow(image.array_to_img(array))
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        for i in range(len(topFMs_pred)):
            sns.heatmap(contribs_pred[i],
                        vmin=v[i][1], vmax=v[i][0], cmap ="OrRd",
                        annot = False, center = (v[i][1]+v[i][0])/2,
                        yticklabels = False, xticklabels = False, ax = axes[i+1],)
            axes[i+1].set_title("feature map "+str(topFMs_pred[i]), fontsize = 10)
        #plt.savefig('matchstick_cond.pdf', bbox_inches = 'tight')
    if rank:
        contribs = np.mean(contribs, axis = (0,1))
        contribs = list(contribs - allClassMeans[layer])
        contribs = [(i, contribs[i]) for i in range(len(contribs))]
        contribs = sorted(contribs, key = lambda x: x[1], reverse = True)
    return contribs, contribs_pred

def get_labels(val_index, img_per_class, pred_path, num_class):
    L = []
    for c in range(num_class):
        for i in range(len(val_index[c])):
            L.append(val_index[c][i][0]*img_per_class + val_index[c][i][1])
    test_pred = pickle.load(open(pred_path, 'rb'))
    ground_truth = np.array([[c for i in range(len(test_pred[c]))] for c in range(num_class)])
    test_pred = np.array([test_pred[c] for c in range(num_class)]).reshape((1,-1))[0]
    ground_truth = ground_truth.reshape((1,-1))[0]
    test_labels = test_pred != ground_truth
    test_labels = test_labels[L]
    return test_labels