import numpy as np
import regressao_linear as rl
import matplotlib.pyplot as plt


def testes():
    x = np.array([
        [1, 3],
        [1, 5],
        [1, 10],
        [1, 12]
    ])
    y = np.array([4, 10, 22, 24]).reshape((1, -1)).T
    w = np.array([1, 1]).reshape((1, -1)).T

    print(np.ones_like(x))

    # print(x.shape, y.shape, w.shape)
    print('Gradiente descendente:')
    print(rl.GD(x, y, w, 0.005, 5000))

    print('Mínimos quadrádicos:')
    print(rl.OLS(x, y))


def pressao():
    dataset = np.genfromtxt('./pressão.txt', delimiter=',', skip_header=1)
    print(dataset.shape)
    n, m = dataset.shape

    x = np.c_[np.ones((n, m-1)), dataset[:, -2].reshape((-1, m-1))]
    y = dataset[:, -1].reshape((-1, 1))
    w = np.array([98, 1]).reshape((1, -1)).T

    print(x.shape, y.shape, w.shape)
    # np.seterr(all='raise')
    w, custos = rl.GD(x, y, w, 0.00001, 130000)
    print(w)
    plt.plot(custos)
    print(rl.gradiente_descendente_estocatico(x, y, w, 0.000000000001, 1))

    w = rl.OLS(x, y)
    print(w)
    print(rl.RMSE(x, y, w))

# pressao()

x = np.c_[np.ones((3, 1)), np.array([1, 2, 3]).reshape((-1, 1))]
y = np.array([2, 4, 6]).reshape((-1, 1))
w = np.array([0, 1]).reshape((1, -1)).T
print(x.shape, y.shape, w.shape)

print(rl.GD(x, y, w, 0.01, 12000)[0])
print(rl.OLS(x, y))
print(rl.SGD(x, y, w, 0.01, 12000))
