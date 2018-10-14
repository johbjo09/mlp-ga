# Perceptron

import numpy as np
from ga import GeneticThing
import sys

class Perceptron():
    def __init__(self, num_inputs, num_targets, eta = 0.05):
        self.num_inputs = num_inputs
        self.num_targets = num_targets

        # m: num_targets
        # n: num_inputs
        # W : m x n matrix,
        
        # Initialize W with gaussians
        self.W = self._get_random_weights()

        self.eta = eta

    def _get_random_weights(self):
        # return (np.random.normal(0, 2, num_in + 1))
        return (np.random.rand(self.num_targets, self.num_inputs+1)-0.5) * 2/np.sqrt(self.num_inputs)

    # Add one set of extra inputs as bias
    def _x1(self, X):
        return np.append(X, [ np.ones(X.shape[1]) ], axis=0)

    def _recall(self, x1):
        # output vector
        o = np.matmul(self.W, x1)
        return 0 + (o > 0)

    def recall(self, X):
        return self._recall(self._x1(X))

    # Delta rule
    def update_weights(self, x1, T):
        # output vector
        o = self._recall(x1)

        # error
        e = np.subtract(o, T)
        
        # Updated weights
        dW = self.eta * np.matmul(np.array(e), np.transpose(x1))

        self.W = self.W - dW

    def train(self, X, T):
        epochs = 20
        x1 = self._x1(X)
        
        for i in range(0, epochs):
            self.update_weights(x1, T)

