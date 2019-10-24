import shutil
import os
import numpy as np
import pickle


num_class = 100

keep = pickle.load(open('keep100.p', 'rb')) # keep100.p is a list with 100 random integers between 0 and 1000 representing
                                            # the 100 randoms classes seleted from Imagenet to create the subset of 
                                            # 100 Classes Imagenet

origin_path = '/data/Datasets/ImageNet/train/'
origin_classes = np.array(sorted(os.listdir(origin_path)))
path = '/data/Datasets/ImageNet/val_small/'
classes = origin_classes[keep]

for c in classes:
    img_folder = sorted(os.listdir(origin_path+c))
    os.mkdir(path+c)
    for i in range(500, 550):
        shutil.copy(origin_path+c+'/'+img_folder[i], path+c+'/'+img_folder[i])
