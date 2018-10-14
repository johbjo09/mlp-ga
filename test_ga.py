
import numpy as np
from copy import deepcopy
from ga import GeneticThing, GeneticAlgorithm
from perceptron import Perceptron
from mlp import MLP

# Multi-layer perceptron

class GeneticMLP(GeneticThing, MLP):

    def __init__(self, num_in):
        MLP.__init__(self, num_in)
        self._fitness = 0

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness

    def mutate(self, r_mutate):
        # Add noise to weights. Noise proportional to 0 < r_mutate < 1
        for i in range(len(self.W)):
            w_shape = np.shape(self.W[i])
            self.W[i] = (1-r_mutate) * self.W[i] + r_mutate * self._get_random_weights(w_shape[0], w_shape[1])

    def crosswith(self, that_p):
        # TODO: random chromosome crossing
        p = deepcopy(self)
        for i in range(len(self.W)):
            p.W[i] = p.W[i] * 0.5 + that_p.W[i] * 0.5
        return p

    def distanceto(self, that_p):
        d = 0
        for i in range(len(self.W)):
            d += np.sum(np.power(self.W[i] - that_p.W[i], 2))
        return d
    
    def add(self, that_p):
        for i in range(len(self.W)):
            self.W[i] += that_p.W[i]

    def divide_by(self, divisor):
        for i in range(len(self.W)):
            self.W[i] /= divisor

# Single layer perceptron

class GeneticPerceptron(GeneticThing, Perceptron):

    def __init__(self, num_inputs, num_targets, eta=0.05):
        Perceptron.__init__(self, num_inputs, num_targets, eta)
        self._fitness = 0

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness

    def mutate(self, r_mutate):
        # Add noise to weights. Noise proportional to 0 < r_mutate < 1
        w_shape = np.shape(self.W)
        self.W = (1-r_mutate) * self.W + r_mutate * self._get_random_weights()

    def crosswith(self, that_p):
        # TODO: random chromosome crossing
        p = deepcopy(self)
        p.W = p.W * 0.5 + that_p.W * 0.5
        return p

    def distanceto(self, that_p):
        return np.sum(np.power(self.W - that_p.W, 2))

    def add(self, that_p):
        self.W += that_p.W

    def divide_by(self, divisor):
        self.W /= divisor

def fitness(T, y):
    return np.sum(1.0 - (T - y))

def test_ga_mlp():
    POP_SIZE = 30
    genetics = GeneticAlgorithm()

    def make_genetic_mlp():
        p = GeneticMLP(2)
        p.add_layer(2)
        p.add_layer(1)
        return p

    X = np.array([ [0,0], [0,1], [1,0], [1,1] ])
    T_xor = np.array([ (x[0] ^ x[1]) for x in X ])
    
    for i in range(POP_SIZE):
        p = make_genetic_mlp()
        p.train(X, T_xor)
        genetics.append(p)
    
    for generation in range(20):
        for p in genetics:
            y = p.recall(X)
            p.set_fitness(fitness(T_xor, y))
            # print "Generation: " + str(generation) + ", " + str(p.fitness)
        genetics.evolve()

def test_ga_perceptron():
    POP_SIZE = 20
    genetics = GeneticAlgorithm()

    for i in range(POP_SIZE):
        genetics.append(GeneticPerceptron(2, 1))

    X = np.array([[ 0, 0 ], [0,1], [1, 0], [1, 1] ])
    T_or = np.array([ (x[0] | x[1]) for x in X ])
    T_and = np.array([ (x[0] & x[1]) for x in X ])
    X = np.transpose(X)
    
    for generation in range(20):
        for p in genetics:
            y = p.recall(X)
            p.set_fitness(fitness(T_and, y))
            # print "Generation: " + str(generation) + ", " + str(p.fitness)
        genetics.evolve()

def test_perceptron():
    or_perceptron = Perceptron(2, 1)
    and_perceptron = Perceptron(2, 1)

    X = np.array([ np.random.randint(0, 2, 2) for i in range(0,10) ])
    T_or = np.array([ (x[0] | x[1]) for x in X ])
    T_and = np.array([ (x[0] & x[1]) for x in X ])
    X = np.transpose(X)
    
    or_perceptron.train(X, T_or)
    and_perceptron.train(X, T_and)

    o_or = or_perceptron.recall(X)    
    o_and = and_perceptron.recall(X)

    print "Or: "
    print T_or
    print o_or
    print "And: "
    print T_and
    print o_and
    
def runtest():
    # test_ga_perceptron()
    test_ga_mlp()

if __name__ == "__main__":
    runtest()
