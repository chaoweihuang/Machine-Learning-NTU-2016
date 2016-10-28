from random import shuffle

import numpy as np

def read_sub_csv(sub):
    l = []
    for line in open(sub):
        if "id,label" in line:
            continue
        line = line.strip().split(',')
        l.append(float(line[1]))
        
    return l


def write_submit_file(l, fname):
    if len(l) != 600:
        return
    with open(fname, 'w') as fw:
        fw.write('id,label\n')
        fw.write('\n'.join(['%d,%d' % (i+1, l[i]) for i in range(600)]))

        
def normalize(l):
    return (l-np.mean(l, axis=0))/np.std(l, axis=0)    

        
def activation(x, _type, deriv=False):
    if _type.lower() == 'relu':
        if deriv:
            return (x > 0).astype(float)
        return np.maximum(x, 0)
    
    if _type.lower() == 'tanh':
        x = np.tanh(x)
        if deriv:
            return 1 - np.power(x, 2)
        return x
    
    if _type.lower() == 'logistic':
        x = np.clip(x, -500, 500)
        x = 1.0 / (1 + np.exp(-x))
        if deriv:
            return np.multiply(x, 1-x)
        return x


def derivatives(a, delta, _type):
    if _type.lower() == 'relu':
        delta[a == 0] = 0
        
    if _type.lower() == 'tanh':
        delta *= (1 - a ** 2)
        
    if _type.lower() == 'logistic':
        delta *= a
        delta *= (1 - a)

        
def expit(x):
    x = np.tanh(x * 0.5)
    x = (x + 1) * 0.5
    return x


def loss(y, y_hat):
    return np.sum(np.absolute(y-y_hat))

    
def accuracy(y, y_hat):
    return 1 - np.mean(np.absolute(y-y_hat))


def read_train_csv(f):
    lists = []
    for line in open(f):
        l = [float(num) for num in line.strip().split(',')[1:]]
        lists.append(l)
    return lists

def get_data(f=None, data=None):
    if data == None:
        data = eval(open(f).read())
    X = np.array([np.array([1.0] + l[:57]) for l in data])
    y = np.array([np.array([l[-1]]) for l in data])
    return X, y

    
def get_splits(X, y, num_cv=5):
    # get cv data
    for cv in range(num_cv):
        indices = np.arange(len(X))
        test_index = np.zeros(len(X)).astype(bool)
        test_index[np.arange(cv,len(X),num_cv)] = True
        train_index = np.logical_not(test_index)
        
        tX, ty = X[train_index], y[train_index]
        cvX, cvy = X[test_index], y[test_index]
        #cv_start, cv_end = int(len(X) / num_cv * cv), int(len(X) / num_cv * (cv+1))
        #tX, ty = np.concatenate((X[:cv_start], X[cv_end:])), np.concatenate((y[:cv_start], y[cv_end:]))
        #cvX, cvy = X[cv_start:cv_end], y[cv_start:cv_end]
        yield tX, ty, cvX, cvy

    
class ADAM:
    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.params = [param for param in params]
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]
        
    def update_params(self, grads):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates

