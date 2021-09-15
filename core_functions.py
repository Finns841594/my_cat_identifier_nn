# -*- coding: utf-8 -*-

'defined the core functions for my DNN '

__author__ = 'Feng Yang'

from abc import ABC
import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    None
    #X = (X - np.mean(X))/!!!!!!!!!!!!!!!!!


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])/ np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
    return parameters


##### FORWARD STARTED #####
###########################

#############
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache
#############

def linear_forward(A,W,b):
    # Z = np.dot(W, A) + b   # we want to keep the Z with shape(n_features, m), so we use W times A
    Z = W.dot(A) + b
    AWb_cache = (A, W, b)
    return Z, AWb_cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation(X, parameters):
    A = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A  #This line helps activations pass to next layer of neural networks
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activation='relu')
        caches.append(cache)

    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation='sigmoid')
    caches.append(cache)

    return AL, caches


##### FORWARD ENDED #####
#########################


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1./m) * (np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


##### BACKWARD STARTED #####
############################

#############
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
#############

def backward_linear(dZ, cache):
    m = len(dZ)
    A_prev, W, b = cache

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True) # 没有乘除的向量化求和需要用sum
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_linear_activation(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide((1-Y), (1-AL))) # This fomula is based on the cost function

    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = backward_linear_activation(dAL, caches[-1], activation='sigmoid')

    for l in reversed(range(L-1)):
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = backward_linear_activation(grads["dA"+str(l+1)], caches[l], activation='relu')

    return grads


##### BACKWARD ENDED #####
##########################


def update_parameters(parameters, grads, learning_rate):
    # parameters = parameters.copy()
    L = len(parameters) // 2 # parameters里包含W和b，所以要除以二

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]

    return parameters
   

def predict(x, y, parameters):
    m = x.shape[1]
    p = np.zeros((1,m))

    pred_y, cache = forward_propagation(x, parameters)

    for i in range(m):
        if pred_y[0, i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: ", str(np.sum((p == y)/m)))

    return p


def print_mislabeled_images(classes, X, y, p):

    a = p + y # if p equals y (predicted correctly), then p+y equals either 0 or 2
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i+1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title('Prediciton: ' + classes[int(p[0,index])].decode('utf-8') + '\n Class: ' + classes[y[0,index]].decode('utf-8'))

