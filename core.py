# -*- coding: utf-8 -*-

'Core code for my DNN_From_Scratch '
'For cat recongation'

__author__ = 'Feng Yang'


#import the data and librarys
import numpy as np

from sigmoid_and_relu import *
from core_functions import *

train_X_raw = open(input("Training set location:"))
train_Y_raw = open(input("Labels location:"))
m = len(train_Y_raw)
layer_dims = [m,4,5,1] # A list of numbers which shows how many neurals in each layer
L = len(layer_dims) # L indicates the number of times of propagation

#process the data
train_X = normalize(train_X_raw)
train_Y = normalize(train_Y_raw)

#initialate parameters
parameters = initialize_parameters(layer_dims)

#forward propagation
Cache, AL = forward_propagation(train_X, parameters)

#compute the cost
cost = compute_cost(AL, train_Y)

#backward propagation
grads = backward_propagation(cost)

#update parameters, print the cost changes
parameters = update_parameters(parameters, grads, iteration_num=100, learning_rate=0.9)

#test on dev set
test_result = forward_propagation(dev_X, parameters)

#tune hyperparameters

#test on test set