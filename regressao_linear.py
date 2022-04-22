import numpy as np


def MSE(x, y, w):
    y2 = x @ w
    return ((y - y2) ** 2).mean()


def RMSE(x, y, w):
    return MSE(x, y, w) ** 0.5


def normalize(x):
    return x / np.linalg.norm(x, axis=-1)[:, np.newaxis]


def GD(x, y, w, alpha, epochs):
    custos = np.zeros(epochs)
    # x = normalize(x)
    for epoch in range(epochs):
        y2 = x @ w
        w = w + alpha * (x.T @ (y - y2))
        custos[epoch] = MSE(x, y, w)

    return w, custos


def SGD(x, y, w, alpha, epochs):
    n, m = x.shape
    custos = np.zeros((epochs * n))
    for epoch in range(epochs):
        for i in range(n):
            y2 = x[i] @ w
            error = (y[i] - y2)
            w = w + alpha * error * x[i].reshape((1, -1)).T
            custos[epoch * n + i] = MSE(x, y, w)

    return w, custos


def OLS(x, y):
    return np.linalg.solve(x.T @ x, x.T @ y)
