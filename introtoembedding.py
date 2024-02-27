import numpy as np
np.random.seed(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

alpha, iterations = (0.01, 2) # What is alpha used for?
hidden_size = 100

weights_0_1 = 0.2*np.random.random((len(vocab)) ,hidden_size)
