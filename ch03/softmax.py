import numpy as np


def softmax(x):
    x = x - np.max(x)   # do this to avoid overflow
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))
