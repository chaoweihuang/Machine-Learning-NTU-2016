import pickle
import numpy as np

def get_data(f):
    data = pickle.load(open(f, 'rb'))

    X, y = [], []
    for cls in range(10):
        for x in data[cls]:
            x = np.asarray(x)
            X.append(x.reshape(3,32,32)[[1,2,0]])
            y.append([cls])

    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

def get_unlabel_data(f):
    data = pickle.load(open(f, 'rb'))

    X = []
    for x in data:
        x = np.asarray(x)
        x = x.ravel().reshape((3, 32, 32))[[1,2,0]]
        X.append(x)

    X = np.asarray(X)
    
    return X

def get_test_data(f):
    data = pickle.load(open(f, 'rb'))

    X = []
    for x in data['data']:
        x = np.asarray(x)
        x = x.ravel().reshape((3, 32, 32))[[1,2,0]]
        X.append(x)

    return np.asarray(X)
