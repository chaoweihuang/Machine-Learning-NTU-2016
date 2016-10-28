import numpy as np
import math

from utility import *

class MLP:
    def __init__(self, activation_type=['relu'], max_passes=20000, reg=0.1, hidden_layer_size=(100,), seed=25639):
        self.W = [0.0]
        self.b = 0.0
        self.activation_type = activation_type
        self.max_passes = max_passes
        self.reg = reg 
        self.hidden_layer_size = hidden_layer_size
        self.seed = seed
        self.dims = (1,) + hidden_layer_size + (1,)
        self.optimizer = None
    
    @staticmethod
    def update_rule(ori, new):
        return ori if new == None else new
        
    def set_params(self, activation_type=None, max_passes=None, reg=None, hidden_layer_size=None, seed=None):
        self.activation_type = update_rule(self.activation_type, activation_type)       # functions for this network
        self.max_passes = update_rule(self.max_passes, max_passes)                      # number of iteration
        self.reg = update_rule(self.reg, reg)                                           # parameters for gradient descent  
        self.hidden_layer_size = update_rule(self.hidden_layer_size, hidden_layer_size) # set dimensions for this network     
        self.seed = update_rule(self.seed, seed)                                        # for random number generator

    def initialize_coef(self, X=[[]], y=[[]]):
        if len(X) == 0 or len(y) == 0:
            print('WARNING: X,y should be arrays with length > 0')
            return
        num_examples = len(X)
        input_dim = len(X[0])
        output_dim = 1 if len(y.shape) == 1 else len(y[0])
        dims = (input_dim,) + self.hidden_layer_size + (output_dim,)
        self.dims = dims
        
        np.random.seed(self.seed)

        # initialize the parameters to random numbers
        self.W = [np.random.uniform(-np.sqrt(2.0/(dims[i]+dims[i+1])), np.sqrt(2.0/(dims[i]+dims[i+1])), (dims[i], dims[i+1])) for i in range(len(dims)-1)]
        self.b = [np.random.uniform(-np.sqrt(2.0/(dims[i]+dims[i+1])), np.sqrt(2.0/(dims[i]+dims[i+1])), dims[i+1]) for i in range(len(dims)-1)]

        # adam optimizer from sklearn
        params = self.W + self.b
        self.optimizer = ADAM(params=params)

    def fit(self, X, y):
        self.initialize_coef(X, y)
        
        num_examples = len(X)
        
        if len(y.shape) == 1:
            y = np.array([[num] for num in y])
        
        # when should we stop?
        last_error = 1000000.0
        consecutive_increase = 0
        
        for num_iter in range(self.max_passes):

            a = self.forward(X)
              
            error = loss(a[-1], y)
            values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.W]))
            error_reg = error + (0.5 * self.reg) * values / num_examples
            
            if num_iter % 50 == 0:
                train_error = np.mean(np.absolute((a[-1]>0.5).astype(float)-y))
                print('\t', num_iter, '\t', error_reg, '\t', train_error)
                
                if error_reg > last_error:
                    consecutive_increase += 1
                elif error_reg < last_error:
                    consecutive_increase = 0
                    
                last_error = error_reg
                
                if consecutive_increase == 2:
                    print('consecutive increasing, stop training')
                    break
            
            deltas = [np.array([]) for i in range(len(self.dims))]    
            dW, db = [0 for i in range(len(self.W))], [0 for i in range(len(self.b))]
            
            last = len(self.dims)-2
            deltas[last] = a[-1] - y
            
            for i in range(len(self.dims)-2, 0, -1):
                deltas[i-1] = np.dot(deltas[i], self.W[i].T)
                derivatives(a[i], deltas[i-1], self.activation_type[i-1])
                dW[i-1] = (np.dot(a[i-1].T, deltas[i-1]) + self.reg * self.W[i-1]) / num_examples
                db[i-1] = np.mean(deltas[i-1], 0)
            
            grads = dW + db
            self.optimizer.update_params(grads)
            
            
    def forward(self, X):
        z, a = [0 for i in range(len(self.dims))], [X]+[0 for i in range(len(self.dims)-1)]
        
        for i in range(len(self.hidden_layer_size)+1):
            z[i+1] = a[i].dot(self.W[i]) + self.b[i]
            if (i + 1) != len(self.hidden_layer_size)+1:
                a[i+1] = activation(z[i+1], self.activation_type[i])
            else:
                a[i+1] = z[i+1]
        
        return a
    
    
    def predict(self, X):
        a = self.forward(X)
        return a[-1]
        
        
    def save(self, filename='neural_net'):
        np.savez(filename, 
                 W=self.W, 
                 b=self.b, 
                 activation_type=self.activation_type, 
                 hidden_layer_size=self.hidden_layer_size, 
                 seed=self.seed,
                 dims=self.dims)
        
    
    def load(self, filename='neural_net'):
        if not '.npz' in filename:
            filename = filename + '.npz'
            
        params = np.load(filename)
        self.W                  = params['W']
        self.b                  = params['b']
        self.activation_type    = params['activation_type']
        self.hidden_layer_size  = params['hidden_layer_size']
        self.seed               = params['seed']
        self.dims               = params['dims']
