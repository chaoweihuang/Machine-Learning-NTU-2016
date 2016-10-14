from scipy.stats import spearmanr, pearsonr

import numpy as np
            
def read_train_csv():
    d = {}
    for line in open("data/train.csv"):
        l = line.strip().split(',')
        if not l[0] in d:
            d[l[0]] = {}
        d[l[0]][l[2]] = l[3:]
        
    return d

def read_test_csv(length):
    d = {}
    if not 1 <= length <= 9:
        return
    start = 11-length
    for line in open("data/test_X.csv"):
        l = line.strip().split(',')
        if not l[0] in d:
            d[l[0]] = {}
        d[l[0]][l[1]] = [i if i != 'NR' else '0.0' for i in l[start:]]
        
    return d

def read_sub_csv(sub):
    l = []
    for line in open(sub):
        if "id,value" in line:
            continue
        line = line.strip().split(',')
        l.append(float(line[1]))
        
    return l

def train_dict_to_list(d, ranges):
    l = []
    for r in ranges:
        for key in d:
            l.append({k: [d[key][k][i] for i in r] for k in d[key]})
            
    return l

def write_submit_file(l, fname):
    if len(l) != 240:
        return
    with open(fname, 'w') as fw:
        fw.write('id,value\n')
        fw.write('\n'.join(['id_%d,%.3f' % (i, l[i]) for i in range(240)]))
        
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
    
    if _type.lower() == 'sigmoid':
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

def get_feature(d):
    avg = eval(open('data/avg.dict').read())
    r = [1.0]
    for k in ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'NMHC', 'NO', 'WD_HR', 'CO']:
        r += [float(num) for num in d[k][:7]] + [float(num)**2/float(avg[k]) for num in d[k][:7]]
    return np.array(r)

def get_ans(d):
    return np.array([int(d['PM2.5'][7])])  

def get_data(f):
    data = eval(open(f).read())
    X = np.array([get_feature(data[i]) for i in range(len(data))])
    y = np.array([get_ans(data[i]) for i in range(len(data))])
    return X, y
    
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

