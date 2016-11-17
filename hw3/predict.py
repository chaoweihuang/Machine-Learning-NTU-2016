from keras.models import load_model

import sys
import os
import numpy as np

from preprocessing import get_test_data
from utility import write_submit_file

data_dir = sys.argv[1]
model_name = sys.argv[2]
output_name = sys.argv[3]
tX = get_test_data(os.path.join(data_dir, 'test.p'))

model = load_model(model_name)

ans = model.predict_classes(np.asarray(tX))
write_submit_file(ans, output_name)
print()
