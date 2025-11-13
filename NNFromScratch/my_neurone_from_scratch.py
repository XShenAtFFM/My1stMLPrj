# implement sigmoid, forward propagation, cost computation, and gradient descent updates.
# Vectorize all computations using NumPy.
# Deliverables: logistic_regression.py, training accuracy, cost plot.


import numpy as np

def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def my_forwardpropagation(x, weights, biases):
    # z =  x * weights + biases
    z = np.dot(weights, x) + biases
    return z

def my_relu(x):
    return np.maximum(x, 0)

def my_logistic_cost(y_pred, y_true):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def my_logistic_gradient(y_hat, y_true):
    eps = 1e-8
    return (y_hat - y_true) / ((y_hat + eps) * (1 - y_hat + eps))


def my_sigmoid_derived(a):
    # a = sigmoid(x)
    # da/dx = a * (1 - a)
    return a * (1 - a)

def my_relu_derived(x):
    # a = relu(x)
    # a = x if x > 0 else 0
    # da/dx = 1 if x > 0 else 0
    return (x > 0).astype(float)

def my_dense_layer(x, weights, biases, activation_function):
    z = my_forwardpropagation(x, weights, biases)
    a = my_sigmoid(z) if activation_function == 'sigmoid' else my_relu(z)
    return a, z








