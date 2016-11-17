from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

K.set_dim_ordering('th')

import numpy as np
import os
import sys

from preprocessing import *


data_dir = sys.argv[1]
model_name = sys.argv[2]

X_train, _ = get_data(os.path.join(data_dir, 'all_label.p'))
X_unlabel = get_unlabel_data(os.path.join(data_dir, 'all_unlabel.p'))
X_test = get_test_data(os.path.join(data_dir, 'test.p'))

X = np.concatenate((X_train, X_unlabel))

X = X.astype('float32') / 255
X_test = X_test.astype('float32') / 255


input_img = Input(shape=(3, 32, 32))
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)

encoder = Model(input=input_img, output=encoded)

adam = Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy')

autoencoder.fit(X, X,
                nb_epoch=20,
                batch_size=32,
                shuffle=True,
                verbose=1,
                validation_data=(X_test, X_test))

encoder.save(model_name)
