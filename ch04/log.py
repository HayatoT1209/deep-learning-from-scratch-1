import numpy as np
import matplotlib.pyplot as plt


def log_func(x):
    delta = 1e-7
    return np.log(x+delta)


if __name__ == '__main__':
    x = np.arange(0.0, 1.0, 0.01)
    y = log_func(x)

    plt.xlabel("x")
    plt.ylabel("log(x)")
    plt.xlim(0.0, 1.0)
    plt.ylim(-5.0, 0.0)

    plt.plot(x, y)
    plt.show()
