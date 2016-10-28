import numpy as np
from utility import *

class LogisticRegression:
    
    def __init__(self, iteration=50000, adagrad=True, rate=1e-1):
        self.w = []
        self.iteration = iteration
        self.adagrad = adagrad
        self.rate = rate    

    def fit(self, X, y):
        if len(y.shape) == 1:
            y = np.array([[num] for num in y])
        
        if len(X) == 0 or len(X) != len(y):
            print('length of X and y not equal')
            return
            
        init_w = np.array([-1.43] + [0.01 for i in range(len(X[0])-1)])

        self.w = init_w
        w_acc = np.array([1e-5 for i in range(len(X[0]))])

        # run several passes
        for i in range(self.iteration+1):
            # compute gradients of w and sum over all training data
            wgrad = self.compute_grad(X, y)
            
            # compute summation of past gradients, for adagrad
            w_acc = w_acc + wgrad**2
            
            # update parameters, using adagrad
            if self.adagrad:
                self.w = self.w - self.rate*(1.0/np.sqrt(w_acc))*wgrad    
            else:
                self.w = self.w - self.rate*wgrad
            
            # compute and print training error/validation error every 1000 pass
            if i % 1000 == 0:
                # E_in(training set error)
                train_ans = (self.predict(X)>0.5).astype(float)
                train_error = np.mean(np.absolute(y[:,0] - train_ans))
                print('iteration %d,\ttrain error: %f' % (i, train_error))

    
    def predict(self, x):
        return expit(np.dot(x, self.w))
        
    
    def save(self, filename='logistic'):
        np.save(filename, self.w)
        
    
    def load(self, filename='logistic'):
        if not '.npy' in filename:
            filename = filename + '.npy'
            
        self.w = np.load(filename)
    
        
    def compute_grad(self, X, y):
        delta = y[:,0] - self.predict(X)
        wgrad = -delta.dot(X)
        return wgrad
