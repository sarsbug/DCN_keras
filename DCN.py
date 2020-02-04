#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:14:34 2020

@author: dongshuai
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from keras.optimizers import SGD
from keras.losses import mean_squared_error

import metrics
from time import time

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

def autoencoder_with_kmeans(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    clustering_input = Input(shape=(dims[-1],), name='clustering_input')

    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=[x,clustering_input], outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder'),clustering_input
    

def loss_wrapper(encoded_X,label_centers,lambd):
    def loss(y_true, y_pred):
        cost_clustering = K.mean(K.square(label_centers-encoded_X),axis=-1)
        cost_reconstruction = K.mean(K.square(y_true-y_pred),axis=-1)
        
        cost = lambd*cost_clustering+cost_reconstruction
        return cost
    return loss

class DCN(object):
    def __init__(self,dims,n_clusters,lambd=0.5,init='glorot_uniform'):
        super(DCN, self).__init__()
       
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.lambd = lambd
        self.autoencoder, self.encoder,self.clustering_input = autoencoder_with_kmeans(self.dims, init=init)

        self.centers = np.zeros((self.n_clusters,self.dims[-1]))
        self.count = 100*np.ones(self.n_clusters,dtype=np.int)
    
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        temp = []
        for i in range(x.shape[0]):
            t = []
            for k in range(self.dims[-1]):
                t.append(0)
            temp.append(t)
        temp = np.array(temp)
        self.autoencoder.fit([x,temp], x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True
            
    def init_centers(self,x,y=None):
        #init self.centers
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(self.encoder.predict(x))
        self.all_pred = kmeans.labels_
        self.centers = kmeans.cluster_centers_

        print('centers-',self.centers)
        if y is not None:
            self.metric(y,self.all_pred)
        
    def compile(self):
        self.autoencoder.compile(optimizer='adam', loss=loss_wrapper(self.encoder.output,self.clustering_input,0.5))

    def fit(self, x,y,epoches,batch_size=256,save_dir='./models'):                    
        m = x_train.shape[0]
        self.count = 100*np.ones(self.n_clusters,dtype=np.int)
        for step in range(epoches):   
            cost = [] #all cost
            
            for batch_index in range(int(m/batch_size)+1):
                X_batch = x[batch_index*batch_size:(batch_index+1)*batch_size,:]
                
                labels_of_centers = self.centers[self.all_pred[batch_index*batch_size:(batch_index+1)*batch_size]]

                c1 = self.autoencoder.train_on_batch([X_batch,labels_of_centers],X_batch)
                cost.append(c1)
                
                reductX = self.encoder.predict(X_batch)
                #update k-means
                self.all_pred[batch_index*batch_size:(batch_index+1)*batch_size], self.centers, self.count = self.batch_km(reductX,self.centers,self.count)
                
            if step%10 == 0:
                reductX = self.encoder.predict(x)
                km_model = KMeans(self.n_clusters,init=self.centers)
                self.all_pred = km_model.fit_predict(reductX)
                self.centers = km_model.cluster_centers_
                
                print('step-',step,' cost:',np.mean(cost))
#                print('centers-',self.centers)
                print('count-',self.count)
                self.metric(y,self.all_pred)
        print('saving model to:', save_dir + '/DCN_model_final.h5')
        self.autoencoder.save_weights(save_dir + '/DCN_model_final.h5')
        
    def batch_km(self,data, center, count):
        """
        Function to perform a KMeans update on a batch of data, center is the
        centroid from last iteration.
    
        """
        N = data.shape[0]
        K = center.shape[0]
    
        # update assignment
        idx = np.zeros(N, dtype=np.int)
        for i in range(N):
            dist = np.inf
            ind = 0
            for j in range(K):
                temp_dist = np.linalg.norm(data[i] - center[j])
                if temp_dist < dist:
                    dist = temp_dist
                    ind = j
            idx[i] = ind
    
        # update centriod
        center_new = center
        for i in range(N):
            c = idx[i]
            count[c] += 1
            eta = 1.0/count[c]
            center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        center_new.astype(np.float32)
        return idx, center_new, count

    def get_centers_and_types_of_points(self,reductX):
        distances = np.abs(reductX - self.centers[:, np.newaxis])
        label_types = np.min(np.argmin(distances, axis=0),axis=1)
        labels_of_centers = self.centers[label_types]
        return labels_of_centers,label_types
        
    def load_weights(self, weights):  # load weights of DEC model
        self.autoencoder.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        reductX = self.encoder.predict(x)
        labels_of_centers,label_types=self.get_centers_and_types_of_points(reductX)
        return label_types
    
    def metric(self,y, y_pred):
        acc = np.round(metrics.acc(y, y_pred), 5)
        nmi = np.round(metrics.nmi(y, y_pred), 5)
        ari = np.round(metrics.ari(y, y_pred), 5)
        print('acc:',acc)
        print('nmi:',nmi)
        print('ari:',ari)

dcn = DCN(dims=[x_train.shape[-1], 500, 500, 2000, 10], n_clusters=10,lambd=0.05)
dcn.compile()
ae_weights='results/temp/ae_weights.h5'
#ae_weights=None

if ae_weights is None:
    pretrain_epochs = 200
    dcn.pretrain(x=x_train,epochs=pretrain_epochs)
else:
    dcn.autoencoder.load_weights(ae_weights)
dcn.init_centers(x_train,y_train)

dcn.fit(x_train,y_train,epoches=200,batch_size=512)
y_pred = dcn.predict(x_test)
print(y_pred)

dcn.metric(y_test,y_pred)
