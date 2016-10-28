import numpy as np
import sys

from utility import *
from neural_net import MLP

def train():
    fname = sys.argv[1]
    output_fname = sys.argv[2]
    
    X, y = get_data(data=read_train_csv(fname))
    model = MLP(activation_type=['logistic', 'logistic', 'logistic'], hidden_layer_size=(30,30,2), seed=15588)
    
    model.fit(X, y)
    
    model.save(output_fname)
    
    
def test():
    model_name = sys.argv[1]
    test_data = sys.argv[2]
    output_fname = sys.argv[3]
    
    model = MLP()
    model.load(model_name)
    
    tX, ty = get_data(data=read_train_csv(test_data))
    ans = (model.predict(tX)>0.5).astype(float)
    
    write_submit_file(ans, output_fname)
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        train()
    elif len(sys.argv) == 4:
        test()
