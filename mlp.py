# Multi layer perceptron

import numpy as np
import sys
import pdb

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

TOL = 1e-2

def square(x):
    return 0 + (x > 0)

def softmax(y):
    normalisers = np.sum(np.exp(y), axis=1) * np.ones((1, np.shape(y)[0]))
    return np.transpose(np.transpose(np.exp(y) / normalisers))

# Sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoidprim(b):
    return b * (1-b);

def tanh(x):
    ex = np.exp(x)
    negex = np.exp(-x)
    return (ex - negex) / (ex + negex)

def tanhprim(g):
    return 1 - np.power(g, 2)

activations = { 'tanh'    : (tanh, tanhprim),
                'sigmoid' : (sigmoid, sigmoidprim),
                'softmax' : (softmax, lambda x: x),
                'square'  : (square, lambda x: 1),
                'linear'  : (lambda x: x, lambda x: 1) }
class MLP:
    def __init__(self, num_in, eta = 0.1, alpha=0.9, activation="sigmoid", output="sigmoid"):

        self.activation = activations[activation][0]
        self.activationprim = activations[activation][1]

        self.output = activations[output][0]
        
        self.num_in = num_in
        self.W = []          # Weights
        self.dW = []         # Delta weights

        self.a = [ None ]    # activations for each layer
        self.eta = eta
        self.alpha = alpha
        self.layers = 0
        
    def _get_random_weights(self, num_in, num_out):
        # return (np.random.normal(0, 2, num_in + 1))
        return (np.random.rand(num_in, num_out) - 0.5) * 10

    def add_layer(self, num_out):
        # Initialize W with gaussians
        num_in = self.num_in
        if len(self.W) > 0:
            layer_shape = np.shape(self.W[-1])
            if len(layer_shape) > 1:
                num_in = layer_shape[1]

        self.W.append(self._get_random_weights(num_in + 1, num_out))
        self.a.append(0)
        self.dW.append(0)
        self.layers += 1

    # Add one set of extra inputs as bias
    def _add1(self, v):
        return np.concatenate((v, -np.ones((np.shape(v)[0], 1)) ),axis=1)

    def _recall(self, x1):        
        a = np.dot(x1, self.W[0]);
        layer = 0
        
        while layer < len(self.W)-1:
            a = self.activation(a)
            a = self._add1(a)
            layer += 1
            self.a[layer] = a
            
            a = np.dot(a, self.W[layer]);

        return self.output(a)

    def recall(self, X):
        y = self._recall(self._add1(X))
        return y

    # Backward
    def update_weights(self, x1, T):
        self.y1 = self._recall(x1)

        # pdb.set_trace()

        layer = len(self.W)-1
        
        # Different types of output neurons
        delta = self.activationprim(self.y1) * (self.y1 - T)

        while layer > 0:
            self.dW[layer] = self.alpha * self.dW[layer] + np.dot(np.transpose(self.a[layer]), delta)
            delta = self.activationprim(self.a[layer]) * np.dot(delta, np.transpose(self.W[layer]))
            delta = delta[:,:-1]
            layer -= 1

        self.dW[layer] = self.alpha * self.dW[layer] + np.dot(np.transpose(x1), delta)

        layer = len(self.W)
        while layer > 0:
            layer -= 1
            self.W[layer] -= self.dW[layer] * self.eta

    def train(self, X, T, epochs=10000):
        x1 = self._add1(X)
        iteration = 0
        error = 1e10

        while error > TOL and iteration < epochs:
            self.update_weights(x1, T)
            iteration += 1

            if iteration % 100 == 0:
                error = 0.5*np.sum(np.power(T - self.y1, 2))
                print "Error: " + str(error)
            
        return iteration

