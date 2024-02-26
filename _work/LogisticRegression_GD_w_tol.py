import logging

import numpy as np
from matplotlib import pyplot as plt

from utils import loss, sigmoid, add_intercept, BaseLG


class LogisticRegression_GD_w_tol(BaseLG):
    def __init__(self, init_method: str = 'zero',
                 lr=0,
                 num_iter: int = 10000,
                 tol: float = 10**-5,
                 fit_interception: bool = True,
                 verbose: bool = True):
        if init_method not in ['zero', 'random']:
            strError = "init_method must be one of ['zero','random']"
            print(strError)
            logging.error(strError)
            raise ValueError(strError)
        self.init_method = init_method
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.fit_interception = fit_interception
        self.verbose = verbose

    def fit(self, X, y):
        if self.fit_interception:
            X = add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        l_prev = 1000.0
        for i in range(self.num_iter):

            h = sigmoid(np.dot(X.astype(float), self.theta.astype(float)))

            loss_ = loss(h, y)
            self.loss_by_iter_.append(loss_)

            if self.tol:
                if l_prev - loss_ < self.tol:
                    return
                l_prev = loss_

            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta = self.theta - self.lr * gradient
            if self.verbose:
                info_str = f'Iteration {i}, loss func {round(loss_, 3)},'
                info_str += f'coeff {np.around(self.theta.astype(float), 3)}'
                logging.info(info_str)

    def loss_visualize(self, filename='graph', str_roc_auc_score=.01):
        print(self.loss_by_iter_)
        print(range(len(self.loss_by_iter_)))
        plt.plot(range(len(self.loss_by_iter_)), self.loss_by_iter_)
        plt.xticks(np.arange(0, self.num_iter, step=self.num_iter / 5))
        plt.xlabel('Numder of iterations')
        plt.xlabel('loss-function')
        plt.title(f'roc-auc score={str_roc_auc_score}')
        plt.savefig(f'./img/{filename}.png')

    def predict_proba(self, X):
        if self.fit_interception:
            X = add_intercept(X)
        return sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return (self.predict_proba(X) >= threshold).astype(int)
