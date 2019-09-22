import shutil
import os
import numpy as np
import pickle


num_class = 50

np.random.seed(0)
rnd = np.arange(1000)
np.random.shuffle(rnd)
rnd = rnd[:num_class]
keep = sorted(rnd)
pickle.dump(keep, open('keep.p', 'wb'))

origin_path = '/data/Datasets/ImageNet/train/'
origin_classes = np.array(sorted(os.listdir(origin_path)))
path = '/data/Datasets/ImageNet/train_small/'
classes = origin_classes[keep]

for c in classes:
    img_folder = sorted(os.listdir(origin_path+c))
    os.mkdir(path+c)
    for i in range(500):
        shutil.copy(origin_path+c+'/'+img_folder[i], path+c+'/'+img_folder[i])
