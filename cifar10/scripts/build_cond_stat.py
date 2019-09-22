# finding the mean conductance across classes with training data
allC_index = [[] for i in range(10)]
for c in range(10):
    for ind in rights_label_index:
        if label[ind] == c:
            allC_index[c].append(ind)
for c in range(10):
    allC_index[c] = allC_index[c][:1000]

    conductance = []
for c in range(10):
    print("class ---- "+str(c))
    class_cond = []
    b = 0
    for ind in allC_index[c]:
        print(b)
        b+=1
        array = x_train[ind]
        cond = layerOneCond(array, c, convLayers, num_class, input_tensor, outputs, sess, sort = True)
        class_cond.append(cond)
    conductance.append(class_cond)


pickle.dump(conductance, open('train_cond.p','wb'))