#insert in mutation val

def get_treshold(model, label, x_test, ok_model):
    pred = np.argmax(model.predict(x_test), axis = 1)
    correct = []
    correct_lab = []
    for i in range(len(label)):
        if label[i] == pred[i]:
            correct.append(x_test[i])
            correct_lab.append(label[i])

    correct = np.array(correct)
    correct_lab = np.array(correct_lab)

    LCR = np.zeros(len(correct))
    for w in ok_model:
        model.set_weights(w)
        guess = np.argmax(model.predict(correct), axis = 1)
        for i in range(len(guess)):
            if guess[i] != correct_lab[i]:
                LCR[i] += 1
    LCR = LCR / len(ok_model)
    mean = np.mean(LCR)
    std = np.std(LCR)
    return mean+std

thresh = get_treshold(model, label, x_test, ok_model)


def decrease_index(all_index, to_exclude):
    L = []
    for ind in all_index:
        if ind not in to_exclude:
            L.append(ind)
    return L


def pred_adv(phi, thresh, alpha, beta, x_test, ok_model):
    model.load_weights('cifar10resnet_weights.h5')
    pred = np.argmax(model.predict(x_test), axis = 1)
    thresh = thresh*phi
    delta = 0.1*thresh
    p1 = thresh - delta
    p0 = thresh + delta
    N = 100
    z = np.zeros(len(x_test))
    adv = np.zeros(len(x_test))
    SPRT = np.zeros(len(x_test))

    rem_ind = np.arange(len(x_test))
    to_exclude = []

    for n in range(N):
        print("---- n="+str(n)+" ----", end = "\r")
        rem_ind = decrease_index(rem_ind, to_exclude)
        to_exclude = []

        #print(rem_ind)
        data = x_test[rem_ind]
        model.set_weights(ok_model[n])
        mutate_pred = np.argmax(model.predict(data), axis = 1)
        for j in range(len(rem_ind)):
            if mutate_pred[j] != pred[rem_ind[j]]:
                #print('LCR happened for img '+str(rem_ind[j]))
                z[rem_ind[j]] += 1
                SPRT[rem_ind[j]] = (p1**z[rem_ind[j]]*(1-p1)**(n-z[rem_ind[j]])) / (p0**z[rem_ind[j]]*(1-p0)**(n-z[rem_ind[j]]))
                #print("SPRT value is "+str(SPRT[rem_ind[j]]))
                if SPRT[rem_ind[j]] <= beta/(1-alpha):
                    #print("AE decision made for img "+str(rem_ind[j]))
                    adv[rem_ind[j]] = 1
                    to_exclude.append(rem_ind[j])
                if SPRT[rem_ind[j]] >= (1-beta)/alpha:
                    print("NAE decision made for img "+str(rem_ind[j]))
                    to_exclude.append(rem_ind[j])
        #print(to_exclude)
    return adv

adv = pred_adv(0.5, 0.2688, 0.1, 0.1, x_test, ok_model)


pickle.dump(adv, open('adv.p','wb'))