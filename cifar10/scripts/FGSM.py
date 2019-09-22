
#from mutation val ipynb

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
def categorical_crossentropy(output, target):
    _EPSILON = 10e-8
    output /= tf.reduce_sum(output,
                        reduction_indices=len(output.get_shape()) - 1,
                        keepdims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_sum(target * tf.log(output), reduction_indices=len(output.get_shape()) - 1)
def to_sign(grad):
    pos = grad >= 0
    neg = -1*(grad < 0)
    sign = pos + neg
    return sign

def FGSM(x, y, model, epsilon = 0.03):
    if len(x.shape) < 4:
        x = np.expand_dims(x,0)
    if len(y.shape) < 2:
        y = np.expand_dims(y,0)
    sess = K.get_session()
    input_tensor = model.input
    output = model.output
    target = tf.placeholder(tf.float32, shape=(None,10))
    loss = categorical_crossentropy(output, target)
    grad = tf.gradients(loss, input_tensor)
    feed_dict = {input_tensor: x, target: y}
    sess_out = sess.run(grad, feed_dict = feed_dict)[0]
    x_adv = x + epsilon*to_sign(sess_out)
    return x_adv


pred = np.argmax(model.predict(x_train), axis = 1)
label = np.argmax(y_train, axis = 1)
right_ind = np.arange(len(pred))[pred==label]


allADV = []
allADV_ind = []
for i in range(len(right_ind)//1000):
    print(i, end = '\r')
    start = i*1000
    end = start + 1000
    current_ind = right_ind[start:end]
    advs = FGSM(x_train[current_ind], y_train[current_ind], model)
    adv_pred = np.argmax(model.predict(advs), axis = 1)
    nor_pred = np.argmax(model.predict(x_train[current_ind]), axis = 1)
    true_adv = adv_pred != nor_pred
    advs = advs[true_adv]
    adv_ind = current_ind[true_adv]
    allADV.append(advs)
    allADV_ind.append(adv_ind)

allADV = np.concatenate(allADV, 0)
allADV_ind = np.concatenate(allADV_ind, 0)

pickle.dump(allADV, open('allADV.p', 'wb'))
pickle.dump(allADV_ind, open('allADV_ind.p', 'wb'))

allADV = pickle.load(open('allADV.p', 'rb'))
allADV_ind = pickle.load(open('allADV_ind.p', 'rb'))

allADV = allADV[:10000]
allADV_ind = allADV_ind[:10000]

# checking if adv conductance is the same as non adv but with contrib in the adv pred, result --> its not the same cond

array = allADV[2]
pred = np.argmax(model.predict(np.expand_dims(array,0)))
cond = oneCond(array, pred, limit = 10)
pred,cond[1]

corresp_array = x_train[allADV_ind[2]]
corresp_pred = np.argmax(model.predict(np.expand_dims(corresp_array,0))) #corresp_pred = 6
corresp_cond = oneCond(corresp_array, corresp_pred, limit = 10)
corresp_pred,corresp_cond[1]