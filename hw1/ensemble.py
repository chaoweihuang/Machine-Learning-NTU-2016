from os import listdir, path
from utility import read_sub_csv, write_submit_file

import numpy as np

l = [np.array(read_sub_csv(path.join('nn_sub', f))) for f in listdir('nn_sub')]
write_submit_file(np.mean(l, axis=0), 'kaggle_best.csv')
