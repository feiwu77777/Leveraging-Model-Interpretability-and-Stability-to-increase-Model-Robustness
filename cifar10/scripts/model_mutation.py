#insert in mutation val

params_num = 0
allWeights = []
        
for i in range(len(model.layers)):
    if "conv" in model.layers[i].name:
        weights = model.layers[i].get_weights()[0]
        params = weights.shape[3]
        allWeights.append((i,np.arange(params_num,params_num+params)))
        params_num += params

#NS_mutation_weights = []

for i in range(100):
    print(i, end = "\r")
    model.load_weights('cifar10resnet_weights.h5')
    gamma = 0.07 # portion of weights to be changed
    for i in range(len(allWeights)):
        weights, bias = model.layers[allWeights[i][0]].get_weights()
        params = weights.shape[3]
        rnd = np.arange(params)
        np.random.shuffle(rnd)
        rnd = rnd[:int(params*gamma)]
        if len(rnd) >= 2:
            firstW = weights[:,:,:,rnd[0]]
            for j in range(len(rnd)-1):
                weights[:,:,:,rnd[j]] = weights[:,:,:,rnd[j+1]]
            weights[:,:,:,rnd[-1]] = firstW
            model.layers[allWeights[i][0]].set_weights([weights, bias])
    res = model.predict(x_test)
    acc = np.argmax(res, axis = 1) == np.argmax(y_test, axis = 1)
    acc = np.mean(acc)
    if acc > 0.73:
        print(acc)
        NS_mutation_weights.append(model.get_weights())

pickle.dump(NS_mutation_weights, open('NS_mutation_weights.p', 'wb'))

train_pred = np.argmax(model.predict(x_train[:10000]), axis = 1)
train_label = np.argmax(y_train[:10000], axis = 1)
train_acc = np.mean(train_pred == train_label)


#NAI
ok_model = []

for i in range(100):
    print(i, end = "\r")
    model.load_weights('cifar10resnet_weights.h5')
    gamma = 0.01
    rnd = np.arange(params_num)
    np.random.shuffle(rnd)
    rnd = rnd[:int(params_num*gamma)]
    rnd = sorted(rnd)
    for i in range(len(allWeights)):
        for num in rnd:
            if num in allWeights[i][1]:
                index = np.argwhere(allWeights[i][1] == num).item()
                w = model.layers[allWeights[i][0]].get_weights()[0]
                b = model.layers[allWeights[i][0]].get_weights()[1]
                w[:,:,:,index] = -1*w[:,:,:,index]
                model.layers[allWeights[i][0]].set_weights([w,b])
    res = model.predict(x_test)
    acc = np.argmax(res, axis = 1) == np.argmax(y_test, axis = 1)
    acc = np.mean(acc)
    if acc > 0.9*train_acc:
        print(acc)
        ok_model.append(model.get_weights())

pickle.dump(ok_model, open('ok_model.p', 'wb'))
ok_model = pickle.load(open('ok_model.p', 'rb'))
len(ok_model)

pickle.dump(ok_model, open('ok_model_train.p', 'wb'))
ok_model = pickle.load(open('ok_model_train.p', 'rb'))
len(ok_model)