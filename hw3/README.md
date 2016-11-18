# HW3 method2

## trained models
__There are two trained models, trained_model and trained_model-dnn__
* trained_model is the encoder  
* trained_model-dnn is a DNN that accepts encoded features as input and classification as output

## training
    bash train.sh $1 $2

* $1: path to data directory
* $2: output model name

**This script will output two models, model and model-dnn**

## testing
    bash test.sh $1 $2 $3

* $1: path to data directory
* $2: input model name(same as $2 in train.sh)
* $3: prediction.csv
