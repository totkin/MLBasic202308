import numpy as np
from scipy.special import expit

class LogisticRegressionGD():
    def __init__(self, lr=0.01, alpha=0.0001, max_iter=1000,
                 eta=1.0, mu=0.5, epsilon=1e-8):
        """
        :param lr: learning rate
        :type lr: float
        :param alpha: regularization parameter
        :type alpha: float
        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param eta: momentum term
        :type eta: float
        :param mu: smoothing term
        :type mu: float
        :param epsilon: stopping criterion
        :type epsilon: float
        """
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta = eta
        self.mu = mu
        self.epsilon = epsilon


    def fit(self, X, y, w=None, b=None):
        """
        Fit the model with the given data.

        :param X: input data, shape (n_samples, n_features)
        :type X: np.array
        :param y: target values, shape (n_samples,)
        :type y: np.array
        :param w: initial weights, shape (n_features,)
        :type w: np.array, optional
        :param b: initial bias
        :type b: float, optional

        :return: self
        :rtype: object
        """

        n_samples = X.shape[0]
        n_features = X.shape[1]

        if w is None:
            w = np.zeros(n_features)
            b = 0.0
        else:
            b = w[-1]
            w = w[:-1]

        for iteration in range(self.