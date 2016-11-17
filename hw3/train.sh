
KERAS_BACKEND=theano THEANO_FLAGS="device=gpu0" python3 autoencoder.py $1 $2
KERAS_BACKEND=theano THEANO_FLAGS="device=gpu0" python3 encoder.py $1 $2

