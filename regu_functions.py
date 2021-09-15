# -*- coding: utf-8 -*-

'defined the core functions for my DNN '

__author__ = 'Feng Yang'

import numpy as np

def initialize_parameters_he(layers_dims):
    # he_initilization mutiplys w with np.sqrt(2/layers_dims[l-1])
    
    # np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = 
        # parameters['b' + str(l)] =
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
        # YOUR CODE ENDS HERE
        
    return parameters


def compute_cost_with_regu_ltwo(AL, Y, parameters, lambd):
    m = Y.shape[1]
    sum_W = 0

    for i in range(len(parameters)//2):
        W = parameters['W'+str(i+1)]
        sum_W = sum_W + np.sum(np.square(W))
    
    cross_entropy_cost = - (1./m) * (np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T))
    L2_regularization_cost = lambd/(2*m) * sum_W
    cost = cross_entropy_cost + L2_regularization_cost

    cost = np.squeeze(cost)

    return cost

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

##### BACKWARD REGUL #####
##########################

def backward_linear_regu(dZ, cache, lambd):
    m = len(dZ)
    A_prev, W, b = cache

    dW = 1./m * np.dot(dZ, A_prev.T) + lambd/m * W
    db = 1./m * np.sum(dZ, axis=1, keepdims=True) # 没有乘除的向量化求和需要用sum
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_linear_activation_regu(dA, cache, lambd, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear_regu(dZ, linear_cache, lambd)
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear_regu(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def backward_propagation_regu(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide((1-Y), (1-AL))) # This fomula is based on the cost function

    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = backward_linear_activation_regu(dAL, caches[-1], lambd, activation='sigmoid')

    for l in reversed(range(L-1)):
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = backward_linear_activation_regu(grads["dA"+str(l+1)], caches[l], lambd, activation='relu')

    return grads


##### DROPOUT  #####
##########################

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

def linear_activation_forward_do(A_prev,W,b,keep_prob,activation):
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = A * D
        A + A / keep_prob

    elif activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation_do(X, parameters, keep_prob):
    A = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A  #This line helps activations pass to next layer of neural networks
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        A, cache = linear_activation_forward_do(A_prev, W, b, keep_prob, activation='relu')
        caches.append(cache)

    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    AL, cache = linear_activation_forward_do(A, W, b, keep_prob, activation='sigmoid')
    caches.append(cache)

    return AL, caches


def backward_linear_activation_regu_do(dA, cache, lambd, keep_prob, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear_regu(dZ, linear_cache, lambd)
        D = np.random.rand(dA_prev.shape[0], dA_prev.shape[1])
        D = (D < keep_prob).astype(int)
        dA_prev = dA_prev * D
        dA_prev = dA_prev / keep_prob
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear_regu(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def backward_propagation_regu_do(AL, Y, caches, lambd, keep_prob):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide((1-Y), (1-AL))) # This fomula is based on the cost function

    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = backward_linear_activation_regu_do(dAL, caches[-1], lambd, keep_prob, activation='sigmoid')

    for l in reversed(range(L-1)):
        grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = backward_linear_activation_regu_do(grads["dA"+str(l+1)], caches[l], lambd, keep_prob, activation='relu')

    return grads