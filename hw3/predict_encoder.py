from keras.models import load_model
from keras import backend as K

K.set_image_dim_ordering('th')

import sys
import os
import numpy as np

from preprocessing import get_test_data
from utility import write_submit_file

data_dir = sys.argv[1]
model_name = sys.argv[2]
output_name = sys.argv[3]

tX = get_test_data(os.path.join(data_dir, 'test.p'))

encoder = load_model(model_name)
dnn = load_model(model_name + '-dnn')

tX = np.asarray(tX).astype('float32') / 255
tX_feature = encoder.predict(tX)

ans = dnn.predict_classes(tX_feature.reshape(tX_feature.shape[0], 256))
write_submit_file(ans, output_name)
print()
