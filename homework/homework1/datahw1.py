import numpy as np

def create_dataset (noise=0.1):
    x = np.expand_dims(np.sort(np.random.rand(100, 1), None), -1)
    y = 5 * x * x + noise * np.random.rand(100, 1)
    X = np.empty((len(x), 3))

    X[:, 0] = 1
    X[:, 1] = x[:, 0]
    X[:, 2] = x[:, 0] ** 2
    return x, y, X
