# implement sigmoid, forward propagation, cost computation, and gradient descent updates.
# Vectorize all computations using NumPy.
# Deliverables: logistic_regression.py, training accuracy, cost plot.

import numpy as np
from sympy.physics.continuum_mechanics.arch import numpy


def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def my_forwardpropagation(x, weights, biases):
    # z =  x * weights + biases
    z = np.dot(x, weights) + biases
    return z

def my_relu(x):
    return x if x > 0 else 0

def my_logistic_costfunction(y_pred, y_true):
    return np.sum(np.multiply(y_true, np.log(y_pred)) + np.multiply((1 - y_true), np.log(1 - y_pred)), axis=1)

def my_sigmoid_derived(a):
    # a = sigmoid(x)
    # da/dx = a * (1 - a)
    return a * (1 - a)

def my_relu_derived(x):
    # a = relu(x)
    # a = x if x > 0 else 0
    # da/dx = 1 if x > 0 else 0
    return 1 if x > 0 else 0

def my_nn(x, weights, biases, activation_function):
    z = my_forwardpropagation(x, weights, biases)
    a = my_sigmoid(z) if activation_function == 'sigmoid' else my_relu(z)
    return a
