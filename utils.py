import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap

from random import randrange


def sigmoid(z):
    return 1 / (1 + np.exp(-z.astype(float)))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def grad_check_sparse(func, x, analytic_grad,
                      num_checks: int = 10,
                      h: float = 10 ** -5,
                      verbose: bool = True):
    """
  sample a few random elements and only return numerical
  in this dimensions.
  """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h  # increment by h
        fxph = func(x)  # evaluate func(x + h)
        x[ix] -= 2 * h  # increment by h
        fxmh = func(x)  # evaluate func(x - h)
        x[ix] += h  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))

        if verbose:
            strMessage = f'numerical: {grad_numerical,} analytic: {grad_analytic}, relative error: {rel_error}'
            print(strMessage)


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_surface(X, y, clf):
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
