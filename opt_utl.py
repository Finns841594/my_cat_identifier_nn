# functions for parameters updating methods

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    permutation = np.random.permutation(m)  # create a list which is the index of the shuffled X
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    inc = mini_batch_size

    num_of_complete_batches = int(np.floor(m/inc))

    for i in range(num_of_complete_batches):
        mini_batch_X = shuffled_X[:, i*inc:(i+1)*inc]
        mini_batch_Y = shuffled_Y[:, i*inc:(i+1)*inc]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m%inc != 0:
        mini_batch_X = shuffled_X[:, num_of_complete_batches*inc:]
        mini_batch_Y = shuffled_Y[:, num_of_complete_batches*inc:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        s["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        s["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))

    return v, s


def update_parameters_with_adam(parameters, grads, t, learning_rate, beta1, beta2, epsilon):
    L = len(parameters) // 2

    v, s = initialize_adam(parameters)
    v_cor = {}
    s_cor = {}

    for l in range(L):
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        s["dW"+str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*np.square(grads["dW"+str(l+1)])
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.square(grads["db"+str(l+1)])

        v_cor["dW"+str(l+1)] = v["dW"+str(l+1)]/(1-beta1**t)
        s_cor["dW"+str(l+1)] = s["dW"+str(l+1)]/(1-beta2**t)
        v_cor["db"+str(l+1)] = v["db"+str(l+1)]/(1-beta1**t)
        s_cor["db"+str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)

        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * v_cor["dW"+str(l+1)] / (np.sqrt(s_cor["dW"+str(l+1)]) + epsilon)
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * v_cor["db"+str(l+1)] / (np.sqrt(s_cor["db"+str(l+1)]) + epsilon)

    return parameters

# below is the function copied from course to load datas for the test
def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y