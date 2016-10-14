import numpy as np
import math

from utility import *

def forward(X):
    z, a = [0 for i in range(len(dims))], [X]+[0 for i in range(len(dims)-1)]
    for i in range(len(hidden_layer_size)+1):
        z[i+1] = a[i].dot(W[i]) + b[i]
        if (i + 1) != len(hidden_layer_size)+1:
            a[i+1] = activation(z[i+1], activation_type)
        else:
            a[i+1] = z[i+1]
    
    return a

def loss(y, y_hat):
    return np.mean(np.power(y-y_hat, 2))/2

# get data
X, y = {}, {}

X['train'], y['train'] = get_data('data/train.i-i+8.list')
X['valid'], y['valid'] = get_data('data/valid.list')
X['test'], y['test'] = get_data('data/test.7.list')

for i in range(6):
    X['train'] = np.concatenate((X['train'], X['test']))
    y['train'] = np.concatenate((y['train'], y['test']))

# set dimensions for this network
hidden_layer_size = (30,10,5)
num_examples = len(X['train'])
input_dim = len(X['train'][0])
output_dim = 1
dims = (input_dim,) + hidden_layer_size + (output_dim,)

# functions for this network
activation_type = 'relu'

# number of iteration
max_passes = 20000
seeds = eval(open('data/neural_net_best_seed.list').read())

# parameters for gradient descent
epsilon = 1e-4
reg = 0.0001

for seed in seeds:
    np.random.seed(seed)
    # initialize the parameters to random numbers
    W = [np.random.uniform(-np.sqrt(6.0/(dims[i]+dims[i+1])), np.sqrt(6.0/(dims[i]+dims[i+1])), (dims[i], dims[i+1])) for i in range(len(dims)-1)]
    b = [np.random.uniform(-np.sqrt(6.0/(dims[i]+dims[i+1])), np.sqrt(6.0/(dims[i]+dims[i+1])), dims[i+1]) for i in range(len(dims)-1)]
    
    init_w, init_b = W, b
    # adam optimizer from sklearn
    params = W + b
    optimizer = ADAM(params=params)

    # when should we stop?
    last_valid = 1000000.0
    consecutive_increase = 0

    last_num = 0
    for num_iter in range(max_passes):

        a = forward(X['train'])
          
        error = loss(a[-1], y['train'])
        values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in W]))
        error_reg = error + (0.5 * reg) * values / num_examples
        
        if num_iter % 50 == 0:
            last_num = num_iter
            test = read_test_csv(7)
            a_test = forward(np.array([get_feature(test['id_%d'%i]) for i in range(240)]))
            ans = a_test[-1]
            write_submit_file(ans, 'nn_sub/nn_sub_%d.csv' % seed)
            
            a_valid = forward(X['valid'])
            valid_error = loss(a_valid[-1], y['valid'])
            print(num_iter, '\t', error, '\t', np.sqrt(2*error), '\t', np.sqrt(2*valid_error))
            if last_valid < valid_error:
                consecutive_increase += 1
            else:
                consecutive_increase = 0
            if consecutive_increase == 2:
                print('consecutive_increase == 2')
                break
            last_valid = valid_error
        
        deltas = [np.array([]) for i in range(len(dims))]    
        dW, db = [0 for i in range(len(W))], [0 for i in range(len(b))]
        
        last = len(dims)-2
        deltas[last] = a[-1] - y['train']

        
        for i in range(len(dims)-2, 0, -1):
            deltas[i-1] = np.dot(deltas[i], W[i].T)
            derivatives(a[i], deltas[i-1], activation_type)
            dW[i-1] = (np.dot(a[i-1].T, deltas[i-1]) + reg * W[i-1]) / num_examples
            db[i-1] = np.mean(deltas[i-1], 0)
        
        grads = dW + db
        optimizer.update_params(grads)
