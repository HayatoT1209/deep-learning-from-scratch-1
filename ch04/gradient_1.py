import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function1(x):
    return 0.01*x**2 + 0.1*x


def tangent_line(f, x):
    a = numerical_diff(f, x)    # 傾き
    print("At x =", x, ", dy/dx =", a)
    b = f(x) - a*x              # 切片
    return lambda t: a*t + b


if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tangent_func = tangent_line(function1, 5)
    y_tangent = tangent_func(x)

    plt.plot(x, y)
    plt.plot(x, y_tangent)
    plt.show()