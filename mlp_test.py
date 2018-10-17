import numpy as np
import mlp

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def gauss_test():

    # Plots
    P_train = np.array([ (12 * np.random.rand(1, 2)[0] - 6) for i in range(0,200) ])
    P_train = np.transpose(P_train)
    X_train = P_train[0]
    Y_train = P_train[1]
    Z_train = np.reshape(np.exp(-(X_train ** 2+ Y_train **2)/10), (-1,1))

    p = mlp.MLP(2, alpha=0.9, eta=0.01, activation="sigmoid")
    p.add_layer(8)
    p.add_layer(4)
    p.add_layer(1)
    P_train = np.transpose(P_train)

    iterations = p.train(P_train, Z_train)

    print "iterations: " + str(iterations)
    
    P_test = np.array([ (12 * np.random.rand(1, 2)[0] - 6) for i in range(0,200) ])
    P_test = np.transpose(P_test)
    X_test = P_test[0]
    Y_test = P_test[1]
    # Z_test = np.exp(-(X_test ** 2+ Y_test **2)/10) - 0.5
    P_test = np.transpose(P_test)

    Z_recall = p.recall(P_test)

    GRID_POINTS = 100
    
    # Surface grid
    g = np.linspace(-6, 6, GRID_POINTS)

    G_X = np.array( [ g for i in range(0,GRID_POINTS) ])
    G_Y = np.array( [ i*np.ones(GRID_POINTS) for i in g ])
    G_Z = np.exp(-(G_Y ** 2+ G_X **2)/10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe.
    ax.plot_wireframe(G_X, G_Y, G_Z, rstride=5, cstride=5, color="#bbaaaa")

    ax.scatter(X_test, Y_test, Z_recall, color="#00ff00")

    plt.show()

def logic_test():
    or_perceptron = mlp.MLP(2, alpha=0.9, eta=0.01)
    or_perceptron.add_layer(2)
    or_perceptron.add_layer(2)
    or_perceptron.add_layer(1)

    and_perceptron = mlp.MLP(2, alpha=0.9, eta=0.01)
    and_perceptron.add_layer(2)
    and_perceptron.add_layer(1)

    xor_perceptron = mlp.MLP(2, alpha=0.9, eta=0.05)
    xor_perceptron.add_layer(2)
    xor_perceptron.add_layer(2)
    xor_perceptron.add_layer(1)

    # X = np.array([ np.random.randint(0, 2, 2) for i in range(0,10) ])
    X = np.array([[ 0, 0 ], [0,1], [1, 0], [1, 1] ])

    T_or = np.reshape(np.array([ (x[0] | x[1]) for x in X ]), (-1,1))
    T_and = np.reshape(np.array([ (x[0] & x[1]) for x in X ]), (-1,1))
    T_xor = np.reshape(np.array([ (x[0] ^ x[1]) for x in X ]), (-1,1))
    
    or_iterations = or_perceptron.train(X, T_or)
    xor_iterations = xor_perceptron.train(X, T_xor)
    and_perceptron.train(X, T_and)
    
    o_or = or_perceptron.recall(X)    
    o_xor = xor_perceptron.recall(X)    
    o_and = and_perceptron.recall(X)

    print "Or (" + str(or_iterations) + " iterations)"
    print T_or
    print o_or
    print "Xor (" + str(xor_iterations) + " iterations)"
    print T_xor
    print o_xor
    #print "And: "
    #print T_and
    #print o_and

def main():
    #logic_test()
    gauss_test()

if __name__ == "__main__":
    main()
