import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap

from random import randrange

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_surface(X, y, clf):
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def sigmoid(z):
    return 1 / (1 + np.exp(-z.astype(float)))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def eval_numerical_gradient(f, x):
    """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        x[ix] += h  # increment by h
        fxh = f(x)  # evalute f(x + h)
        x[ix] -= h  # restore to previous value (very important!)

        # compute the partial derivative
        grad[ix] = (fxh - fx) / h  # the slope
        print(ix, grad[ix])
        it.iternext()  # step to next dimension
    return grad


def grad_check_sparse(func, x, analytic_grad, num_checks, verbose: bool = True):
    """
  sample a few random elements and only return numerical
  in this dimensions.
  """
    h = 1e-5

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h  # increment by h
        fxph = func(x)  # evaluate f(x + h)
        x[ix] -= 2 * h  # increment by h
        fxmh = func(x)  # evaluate f(x - h)
        x[ix] += h  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))

        if verbose:
            strMessage = f'numerical: {grad_numerical,} analytic: {grad_analytic}, relative error: {rel_error}'
            print(strMessage)
