import logging
from pathlib import Path

from abc import ABC

import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt

from _work.experiments.exp08.dmia.utils import sigmoid

strLogFileName = Path(__file__).name
logging.basicConfig(level=logging.INFO,
                    filename=f"logs/{strLogFileName}.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")


class BaseLG(ABC):
    penalty: str = 'l2'  # 'l2' | 'l1' ## Метод регуляризации

    lr: float = .1  ## Темп обучения
    tol: float = 10 ** -5  ## Допуск сходимости
    max_iter: int = 10 ** 7  ## Максимальное количество итераций градиентного спуска
    _lambda: float = 10 ** -3  ## Сила регуляризации (штрафной коэффициент)
    fit_intercept: bool = True  ## Добавление константы
    verbose: bool = True
    beta: np.ndarray
    _loss_by_iter = []  ## Список значений функции потерь по итерациям
    num_iter: int = 0  ## Количество фактически совершенных итераций


class RegLogReg(BaseLG):

    def __init__(self,
                 penalty='l2',
                 lr=0.1,
                 tol=10 ** -5,
                 max_iter=10 ** 7,
                 _lambda=10 ** -3,
                 fit_intercept=True):

        if penalty not in ['l2', 'l1']:
            strError = "penalty must be 'l1' or 'l2'"
            if self.verbose:
                logging.error(strError)
                print(strError)
            raise ValueError(strError)

        self.penalty = penalty
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self._lambda = _lambda
        self.fit_intercept = fit_intercept
        self.beta = np.array([])

    def fit(self, X, y):
        '''
        Обучение
        '''
        if self.fit_intercept:
            # добавляем константу, т.е. первый столбец из единиц
            X = np.c_[np.ones(X.shape[0]), X]

        # инициализация функции потерь и весов признаков
        l_prev = np.inf
        if not self.beta.any():
            self.beta = np.zeros(X.shape[1])

        # реализация градиентного спуска
        for self.num_iter in range(int(self.max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss, grad = self.logregloss(X, y, y_pred)
            if l_prev - loss < self.tol:
                return
            l_prev = loss
            self._loss_by_iter.append(loss)
            self.beta = self.beta - self.lr * grad
            if self.verbose:
                info_str = f'Iteration {self.num_iter}, loss {round(l_prev, 3)}, beta {np.round(self.beta.astype(float), 3)}'
                print(info_str)
                logging.info(info_str)

    def logregloss(self, X, y, y_pred):

        dim = X.shape[0]
        if not self.beta.any():
            self.beta = np.zeros(X.shape[1])
        print('X.shape[0]', X.shape[0])

        # расчет потерь
        nll = -np.log(np.where(y == 1, y_pred, 1 - y_pred)).sum()
        # сильно оптимизированный код
        param = 2 if self.penalty == 'l2' else 1  # проверка на значения проведена при инициализации
        local_beta = self.beta[1:] if self.fit_intercept else self.beta
        penalty = (self._lambda / param) * np.linalg.norm(local_beta, ord=param) ** param
        loss = (penalty + nll) / dim

        print('loss', loss)

        # градиент
        if self.penalty == 'l2':
            if self.fit_intercept:
                d_penalty = self._lambda * self.beta[1:]
                print(d_penalty, self._lambda, self.beta[1:])
                d_penalty = np.r_[self.beta[0], d_penalty]
            else:
                d_penalty = self._lambda * self.beta

        else:  # self.penalty == 'l1':
            if self.fit_intercept:
                d_penalty = self._lambda * np.sign(self.beta[1:])
                d_penalty = np.r_[self.beta[0], d_penalty]
            else:
                d_penalty = self._lambda * np.sign(self.beta)

        print('y', y.shape, 'y_pred', y_pred, X.shape, 'd_penalty', d_penalty.shape)
        print('(y_pred - y).shape', (y_pred - y).shape)
        print('(y - y_pred).dot(X) ', ((y - y_pred).dot(X)).shape)
        grad = -((y - y_pred).dot(X) + d_penalty) / dim

        return loss, grad

    def _NLL(self, X, y, y_pred):
        np.seterr(divide='ignore')
        nll = -np.log(np.where(y == 1, y_pred, 1 - y_pred)).sum()
        np.seterr(divide='warn')

        print(nll)

        if self.penalty == 'l2':
            if self.fit_intercept:
                penalty = (self._lambda / 2) * np.linalg.norm(self.beta[1:], ord=2) ** 2
                print('\nself._lambda: ', self._lambda,
                      '\nself.beta[1:]: ', self.beta[1:],
                      '\nnp.linalg.norm(self.beta[1:], ord=2) ** 2: ', np.linalg.norm(self.beta[1:], ord=2) ** 2)
                d_penalty = self._lambda * self.beta[1:]
                print('\nd_penalty: ', d_penalty)
                d_penalty = np.r_[self.beta[0], d_penalty]
                print('\nnp.r_[self.beta[0], d_penalty]: ', np.r_[self.beta[0], d_penalty])

            else:
                penalty = (self._lambda / 2) * np.linalg.norm(self.beta, ord=2) ** 2
                d_penalty = self._lambda * self.beta

        if self.penalty == 'l1':
            if self.fit_intercept:
                penalty = self._lambda * np.linalg.norm(self.beta[1:], ord=1)
                d_penalty = self._lambda * np.sign(self.beta[1:])
                d_penalty = np.r_[self.beta[0], d_penalty]
            else:
                penalty = self._lambda * np.linalg.norm(self.beta, ord=1)
                d_penalty = self._lambda * np.sign(self.beta)

        print('\nnp.dot(y - y_pred, X):', np.dot(y - y_pred, X).shape,
              '\nd_penalty', d_penalty.shape,
              '\nX.shape[0]', X.shape[0])

        return (penalty + nll) / X.shape[0], -(np.dot(y - y_pred, X) + d_penalty) / X.shape[0]

    def predict_proba(self, X):
        '''
        вычисление вероятности
        '''
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))

    def predict(self, X, threshold):
        '''
        вычисление прогноза
        '''
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss_visualize(self, filename='graph', str_roc_auc_score=.01):
        print(f'AUC-ROC on test sample:{str_roc_auc_score}')
        plt.plot(range(len(self._loss_by_iter)), self._loss_by_iter)
        plt.xticks(np.arange(0, np.round(self.num_iter, -2), step=50))
        plt.xlabel('Number of iterations')
        plt.ylabel('loss-function')
        plt.title(f'roc-auc score={str_roc_auc_score} on {self.num_iter} iterations')
        plt.savefig(f'./assets/{filename}.png')

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
