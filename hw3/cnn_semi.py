from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

import pickle
import numpy as np
import os
import sys

from preprocessing import *

data_dir = sys.argv[1]
model_name = sys.argv[2]

batch_size = 32
nb_classes = 10
nb_epoch = 100
enhance_iter = 5
confidence_percentage = 0.93
data_augmentation = True

img_rows, img_cols = 32, 32
img_channels = 3

np.random.seed(5487)

X, y = get_data(os.path.join(data_dir, 'all_label.p'))
X_unlabel = get_unlabel_data(os.path.join('all_unlabel.p'))

idx = np.random.permutation(X.shape[0])
X_train = X[idx[:4000]]
y_train = y[idx[:4000]]
X_test  = X[idx[4000:]]
y_test  = y[idx[4000:]]

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',
                        input_shape=X_train.shape[1:]))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, init='he_normal'))
model.add(Activation('relu'))
model.add(Dense(512, init='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes, init='he_normal'))
model.add(Activation('softmax'))

adam = Adam(lr=0.0003)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model_json = model.to_json()

X_train = X_train.astype('float32')
X_unlabel = X_unlabel.astype('float32')
X_test = X_test.astype('float32')


for num_iter in range(enhance_iter):

    if num_iter != 0:
        proba = model.predict_proba(X_unlabel)
        ans = np.argmax(proba, axis=1)
        count_class = [[] for i in range(nb_classes)]
        
        for i in range(proba.shape[0]):
            count_class[ans[i]].append([i, np.max(proba[i])])
        for i in range(nb_classes):
            count_class[i] = np.asarray(sorted(count_class[i], key=lambda x:x[1]))
        
        index = np.array([])
        start_id = int((1 - confidence_percentage) * np.min([len(count_class[i]) for i in range(nb_classes)]))
        for i in range(nb_classes):
            index = np.append(index, count_class[i][-start_id:,0].astype(int))
            
        slice_ = np.zeros(len(X_unlabel)).astype(bool)
        slice_[list(index.astype(int))] = True
        slice_not = np.logical_not(slice_)

        Y_unlabel = np_utils.to_categorical(ans[slice_], nb_classes)
        print(np.unique(ans[slice_], return_counts=True)) 
        X_train = np.concatenate((X_train, X_unlabel[slice_]))
        Y_train = np.concatenate((Y_train, Y_unlabel))

        X_unlabel = X_unlabel[slice_not]
        
        model = model_from_json(model_json)
        
        adam = Adam(lr=0.0003)
        model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    
    print('Enhancement iteration %d:' % num_iter)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
        
    model_checkpoint = ModelCheckpoint(model_name,
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=False, 
                                        mode='auto')
    
    callbacks = [model_checkpoint]

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  verbose=1,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False)

        datagen.fit(X_train)

        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            verbose=1,
                            callbacks=callbacks)

