train_pred = np.argmax(model.predict(x_train), axis = 1)
train_label = np.argmax(y_train, axis = 1)
w = train_pred != train_label
wrongs_index = np.arange(x_train.shape[0])[w]



# constructing adversarial examples
ae = pickle.load(open('allADV.p','rb'))
ae_ind = pickle.load(open('allADV_ind.p','rb'))
ae = ae[:10000]
ae_ind = ae_ind[:10000]

input_tensor, outputs, sess = layerInit(model, num_class, convLayers, True)
wrongs_ae = {}
origin_ind = {}
for i in range(len(ae)):
    array = ae[i]
    pred = np.argmax(model.predict(np.expand_dims(array,0)))
    print(i, end = "\r")
    final_contrib = layerOneCond(array, pred, convLayers, num_class, input_tensor, outputs, sess, sort = False) # len = channel number
    if pred not in wrongs_ae.keys():
        wrongs_ae[pred] = []
        origin_ind[pred] = []
        origin_ind[pred].append(i)
        wrongs_ae[pred].append(final_contrib)
    else:
        origin_ind[pred].append(i)
        wrongs_ae[pred].append(final_contrib)

L = []
for c in range(num_class):
    L.append(origin_ind[c])
L = np.concatenate(L, axis = 0)

allADV_LCR = pickle.load(open('allADV_LCR.p', 'rb'))
allADV_LCR = allADV_LCR[L]
pickle.dump(allADV_LCR, open('allADV_LCR.p', 'wb'))

pickle.dump(wrongs_ae, open("wrong_train_ae_cond.p", "wb"))

wrongs_ae = pickle.load(open("wrong_train_ae_cond.p", "rb"))

wrongs_ae[6] = wrongs_ae[6][:1000] # to not overfit on the 6th class

r = train_pred == train_label
rights_index = np.arange(x_train.shape[0])[r]


input_tensor, outputs, sess = layerInit(model, num_class, convLayers, True)
origin_ind = {}
rights = {}
counts = [4921, 1181, 4072, 3704, 8294, 8630, 19983, 1842, 5869, 8303]
for c in range(num_class):
    count = counts[c]
    origin_ind[c] = []
    rights[c] = []
    while len(origin_ind[c]) != len(wrongs_ae[c]):
        if count == len(rights_index):
            break
        ind = rights_index[count]
        if train_label[ind] == c:
            array = x_train[ind]
            pred = c
            final_contrib = layerOneCond(array, pred, convLayers, num_class, input_tensor, outputs, sess, sort = False)
            print(len(origin_ind[c]), end = "\r")
            origin_ind[pred].append(ind)
            rights[pred].append(final_contrib)
        count += 1


pickle.dump(rights, open("right_train_ae_cond.p", "wb"))

rights_ae = pickle.load(open("right_train_ae_cond.p", "rb"))

L = []
for c in range(num_class):
    L.append(origin_ind[c])
L = np.concatenate(L, axis = 0)


right_train = x_train[L]


ok_model = pickle.load(open('ok_model_train.p', 'rb'))
def calc_LCR(model, x, ok_model):
    model.load_weights('cifar10resnet_weights.h5')
    pred = np.argmax(model.predict(x), axis = 1)
    LCR = np.zeros(len(x))
    for n in range(len(ok_model)):
        print("---- n="+str(n)+" ----", end = "\r")
        model.set_weights(ok_model[n])
        mutate_pred = np.argmax(model.predict(x), axis = 1)
        ye = mutate_pred != pred
        LCR[ye] += 1
    LCR = LCR/len(ok_model)
    return LCR

right_allADV_LCR = calc_LCR(model, right_train, ok_model)

pickle.dump(right_allADV_LCR, open('right_allADV_LCR.p', 'wb'))

