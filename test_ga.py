
import threading
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation

from ga import GeneticThing, GeneticAlgorithm
from perceptron import Perceptron
from mlp import MLP

# Genetic multi-layer perceptron

class GeneticMLP(GeneticThing, MLP):

    def __init__(self, num_in, r_mutation=0.4, severity=5, activation="tanh"):
        MLP.__init__(self, num_in, activation=activation)
        self._fitness = 0
        self._r_mutation = r_mutation
        self.severity = severity

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness

    def mutate(self, r_mutate):
        # Add noise to weights. Noise proportional to 0 < r_mutate < 1
        for i in range(len(self.W)):
            self.W[i] = (1 - self._r_mutation) * self.W[i] + self._r_mutation * self._get_randomization(self.W[i])

    def _get_randomization(self, w):
        # return (np.random.normal(0, 2, num_in + 1))
        w_shape = np.shape(w)
        return (np.random.rand(w_shape[0], w_shape[1]) - 0.5) * self.severity

    def crosswith(self, that_p):
        p = deepcopy(self)
        for i in range(len(self.W)):
            w_shape = np.shape(p.W[i])
            mutations = int(self._r_mutation * w_shape[0] * w_shape[1])
            for j in range(mutations):
                k = 0 if w_shape[0] == 1 else random.randint(1, w_shape[0] -1)
                l = 0 if w_shape[1] == 1 else random.randint(1, w_shape[1] -1)
                p.W[i][k][l] = that_p.W[i][k][l]
                # p.W[i] = p.W[i] * 0.5 + that_p.W[i] * 0.5
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
        p = deepcopy(self)
        # p.W = p.W * 0.5 + that_p.W * 0.5
        mutations = 3
        for j in range(mutations):
            w_shape = np.shape(p.W)
            k = 0 if w_shape[0] == 1 else random.randint(1, w_shape[0] -1)
            l = 0 if w_shape[1] == 1 else random.randint(1, w_shape[1] -1)
            p.W[k][l] = that_p.W[k][l]
        return p

    def distanceto(self, that_p):
        return np.sum(np.power(self.W - that_p.W, 2))

    def add(self, that_p):
        self.W += that_p.W

    def divide_by(self, divisor):
        self.W /= divisor

def fitness(T, y):
    return 1.0 / (np.sum(np.power(T - y, 2) + 0.01))

class MonitorGraph():
    def __init__(self, ax, X_test, Y_test, Z_recall):
        self.ax = ax
        self.generation = 0
        self.X_test = X_test
        self.Y_test = Y_test
        self.Z_recall = Z_recall

        GRID_POINTS = 100
        # Surface grid
        g = np.linspace(-6, 6, GRID_POINTS)
        self.G_X = np.array( [ g for i in range(0,GRID_POINTS) ])
        self.G_Y = np.array( [ i*np.ones(GRID_POINTS) for i in g ])
        self.G_Z = np.exp(-(self.G_Y ** 2+ self.G_X **2)/10)

        # wireframe = ax.plot_wireframe(self.G_X, self.G_Y, self.G_Z, rstride=5, cstride=5, color="#aa7777")
        # self.scatter = self.ax.scatter(self.X_test, self.Y_test, self.Z_recall, color="#00ff00")

    def update(self, frame):
        self.scatter = self.ax.scatter(self.X_test, self.Y_test, self.Z_recall, color="#00ff00")
        self.ax.clear()
        self.ax.set_title('Evolving NN, gen: ' + str(self.generation))
        self.wireframe = self.ax.plot_wireframe(self.G_X, self.G_Y, self.G_Z, rstride=5, cstride=5, color="#aa7777")
        self.wireframe.set_alpha(0.3)
        self.scatter = self.ax.scatter(self.X_test, self.Y_test, self.Z_recall, color="#00ff00")

        # self.scatter.set_data(self.X_test, self.Y_test, self.Z_recall)
        # self.scatter.set_data(self.X_test, self.Y_test, self.Z_recall)
        # data = np.hstack((x[:i,np.newaxis], y[:i, np.newaxis]))
        # scat.set_offsets(data)


def test_ga_gauss():
    POP_SIZE = 40
    genetics = GeneticAlgorithm()

    P_test = np.array([ (12 * np.random.rand(1, 2)[0] - 6) for i in range(0,100) ])
    P_test = np.transpose(P_test)
    X_test = P_test[0]
    Y_test = P_test[1]
    Z_test = np.reshape(np.exp(-(X_test ** 2+ Y_test **2)/10), (-1,1))
    P_test = np.transpose(P_test)

    Z_recall = deepcopy(Z_test)
    
    for i in range(POP_SIZE):
        p = GeneticMLP(2, r_mutation=0.5, severity=10, activation="tanh")
        p.add_layer(5)
        p.add_layer(1)
        #if i % 5 == 0:
        #   p.train(P_test, Z_test)
        genetics.append(p)

    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    monitor = MonitorGraph(ax, X_test, Y_test, Z_recall)

    def evolve():
        for generation in range(1000):
            max_fit = 0
            for p in genetics:
                y = p.recall(P_test)
                fit = fitness(Z_test, y)
                p.set_fitness(fit)
                if fit > max_fit:
                    max_fit = fit
                    monitor.Z_recall = y
                    monitor.generation = generation
            genetics.evolve()

    thread = threading.Thread(target = evolve)
    thread.start()
    
    anim = FuncAnimation(fig, monitor.update, interval=1000)
    plt.show()


def test_ga_xor():
    POP_SIZE = 50
    genetics = GeneticAlgorithm()

    X = np.array([ [0,0], [0,1], [1,0], [1,1] ])
    T_xor = np.reshape(np.array([ (x[0] ^ x[1]) for x in X ]), (-1,1))
    
    for i in range(POP_SIZE):
        p = GeneticMLP(2, r_mutation=0.5, severity=10)
        p.add_layer(2)
        p.add_layer(1)
        # if i % 2 == 0:
        # p.train(X, T_xor, epochs=1000)
        genetics.append(p)
    
    for generation in range(10000):
        for p in genetics:
            y = p.recall(X)
            p.set_fitness(fitness(T_xor, y))
            # print "Generation: " + str(generation) + ", " + str(p.fitness)
        genetics.evolve()

def test_ga_perceptron():
    POP_SIZE = 20
    genetics = GeneticAlgorithm()

    X = np.array([[ 0, 0 ], [0,1], [1, 0], [1, 1] ])
    T_or = np.array([ (x[0] | x[1]) for x in X ])
    T_and = np.array([ (x[0] & x[1]) for x in X ])
    X = np.transpose(X)

    for i in range(POP_SIZE):
        p = GeneticPerceptron(2, 1)
        genetics.append(p)
    
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
    # test_ga_xor()
    test_ga_gauss()
    
if __name__ == "__main__":
    runtest()
