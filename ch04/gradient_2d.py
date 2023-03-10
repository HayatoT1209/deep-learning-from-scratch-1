import numpy as np
import matplotlib.pyplot as plt


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形の勾配行列を作成

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f(x)

        x[idx] = tmp - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

def function2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    a = numerical_gradient(f, x)    # 傾き
    print("At x =", x, ", dy/dx =", a)
    b = f(x) - a*x              # 切片
    return lambda t: a*t + b


if __name__ == '__main__':
    x0 = np.arange(-3.0, 3.0, 0.1)
    x1 = np.arange(-3.0, 3.0, 0.1)
    X, Y = np.meshgrid(x0, x1)

    print("X: ", X)
    print("Y: ", Y)

    X = X.flatten()
    Y = Y.flatten()

    print("X: ", X)
    print("Y: ", Y)

    grad = numerical_gradient(function2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()