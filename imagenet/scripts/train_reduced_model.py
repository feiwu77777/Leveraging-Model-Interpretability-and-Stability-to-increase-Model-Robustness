from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import argparse


parser = argparse.ArgumentParser(description='parser to get the value of num_class')
parser.add_argument("--num_class", default = 100, help = 'can be either 50 or 100')
parser.add_argument("--epochs", default = 2, help = 'can be either 50 or 100')
args = parser.parse_args()

num_class = args.num_class
train_path = 'Datasets/train{}/'.format(num_class)
val_path = 'Datasets/val{}/'.format(num_class)

bs = 10
sz = 299

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True)
train_gen = train_datagen.flow_from_directory(train_path,
                                              target_size=(sz,sz),
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=bs)

valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
valid_gen = valid_datagen.flow_from_directory(val_path,
                                              target_size=(sz,sz),
                                              class_mode='categorical',
                                              shuffle=False,
                                              batch_size=bs)

net = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (sz,sz,3))
model = create_net(None, num_class)

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_gen,
	                 steps_per_epoch = train_gen.n//train_gen.batch_size,
	                 epochs=args.epochs,
	                 validation_data=valid_gen,
	                 validation_steps = valid_gen.n//valid_gen.batch_size)

model.save('model/reduced_weights{}.h5'.format(num_class))
