from keras.datasets import mnist
import numpy as np 

np.random.seed(10)

from time import time
import keras.backend as K 
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model  
from keras.optimizers import SGD 
from keras import callbacks 
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans 
import metrics 

def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0], ), name='input')
    x = input_img 
    # internal layers in encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
    # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[0], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
    
    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)

n_cluster = len(np.unique(y))
# print(x.shape)

kmeans = KMeans(n_clusters=n_cluster, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(x)

acc = metrics.acc(y, y_pred_kmeans)
print(acc)

dims = [x.shape[-1], 500, 500, 2000, 10]
init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 300
batch_size = 256
save_dir = './result'

autoencoder, encoder = autoencoder(dims, init=init)

autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
autoencoder.save_weights(save_dir + '/ae_weights.h5')