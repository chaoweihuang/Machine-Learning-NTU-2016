import math
import numpy as np
from utility import *

def func(w, x):
    return np.dot(w, x)
    
def func_grad(w, x, y):
    diff = y[0]-func(w,x)
    wgrad = np.array([2*(-x[i])*diff for i in range(len(x))])
    return wgrad

feature_train, ans_train = get_data('data/train.i-i+8.list')
feature_test, ans_test = get_data('data/test.7.list')
feature_valid, ans_valid = get_data('data/valid.list')

for i in range(10):
    feature_train = np.concatenate((feature_train, feature_test))
    ans_train = np.concatenate((ans_train, ans_test))

init_w = np.array([-2.5] + [0.0 for i in range(len(feature_train[0])-1)])

#w = init_w
w = eval(open('data/simple_linear_best_w.list').read())
iteration = 100
rate = 1e-2

w_acc = np.array([0.0 for i in range(len(feature_train[0]))])

# run several passes
for i in range(iteration+1):
    # compute gradients of w and sum over all training data
    wgrad = sum(func_grad(w, feature_train[j], ans_train[j]) for j in range(len(feature_train)))
    
    # compute summation of past gradients, for adagrad
    w_acc = w_acc + wgrad**2
    
    # update parameters, using adagrad
    w = w - rate*(1.0/(w_acc)**0.5)*wgrad
    #w = w - rate*wgrad    
    # compute and print training error/validation error every 1 pass
    if i % 1 == 0:
        # E_in(training set error)
        train_ans = np.dot(feature_train, w)
        # use RMSE as error measurement
        train_error = np.sqrt(np.mean((ans_train[:,0] - train_ans)**2))
        
        # validation set error
        valid_ans = np.dot(feature_valid, w)
        valid_error = np.sqrt(np.mean((ans_valid[:,0] - valid_ans)**2))
        print('iteration %d,\ttrain error: %f,\t valid error: %f' % (i, train_error, valid_error))
        
test = read_test_csv(7)
ans = [func(w, get_feature(test['id_%d' % i])) for i in range(240)]
write_submit_file(ans, 'linear_regression.csv')
