from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

num_class = 50
bs = 10
sz = 299

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True)
train_gen = train_datagen.flow_from_directory('/data/Datasets/ImageNet/train_small/',
                                              target_size=(sz,sz),
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=bs)

valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
valid_gen = valid_datagen.flow_from_directory('/data/Datasets/ImageNet/val_small/',
                                              target_size=(sz,sz),
                                              class_mode='categorical',
                                              shuffle=False,
                                              batch_size=bs)

net = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (299,299,3))
for layer in net.layers:
    layer.trainable = False

x = net.output
x = GlobalAveragePooling2D()(x)
x = Dense(num_class)(x)
outputs = Activation('softmax')(x)
model = Model(inputs = net.input, outputs = outputs)

opt = Adam(lr=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_gen,
	                 steps_per_epoch = train_gen.n//train_gen.batch_size,
	                 epochs=2,
	                 validation_data=valid_gen,
	                 validation_steps = valid_gen.n//valid_gen.batch_size)

model.save('reduced_weights.h5')
