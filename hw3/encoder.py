from keras.layers import Dense, Activation
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

import numpy as np
import os
import sys

from preprocessing import *

data_dir = sys.argv[1]
model_name = sys.argv[2]
nb_classes = 10

X_train, y_train = get_data(os.path.join(data_dir, 'all_label.p'))
X_test = get_test_data(os.path.join(data_dir, 'test.p'))

Y_train = np_utils.to_categorical(y_train, nb_classes)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

idx = np.random.permutation(X_train.shape[0])
X_valid = X_train[idx[4500:]]
Y_valid = Y_train[idx[4500:]]
X = X_train[idx[:4500]]
Y = Y_train[idx[:4500]]

encoder = load_model(model_name)
X_feature = encoder.predict(X)
X_valid_feature = encoder.predict(X_valid)

X_feature = X_feature.reshape(X_feature.shape[0], 256)
X_valid_feature = X_valid_feature.reshape(X_valid_feature.shape[0], 256)

model = Sequential()
model.add(BatchNormalization(input_shape=X_feature.shape[1:]))
model.add(Dense(1024, init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(512, init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(nb_classes, init='he_normal'))
model.add(Activation('softmax'))

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_feature, Y,
                nb_epoch=50,
                batch_size=32,
                shuffle=True,
                verbose=1,
                validation_data=(X_valid_feature, Y_valid))

model.save(model_name + '-dnn')
