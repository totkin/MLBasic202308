import logging
from pathlib import Path

from abc import ABC

import numpy as np

strLogFileName = Path(__file__).name
logging.basicConfig(level=logging.INFO,
                    filename=f"logs/{strLogFileName}.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


class BaseLG(ABC):
    penalty: str = 'l2'  # 'l2' | 'l1' ## Метод регуляризации
    # градиентный спуск не эффективен для L1-регуляризации, оставлено под последующую доработку

    lr: float = .1  ## Темп обучения
    tol: float = 10 ** -5  ## Допуск сходимости
    max_iter: int = 10 ** 7  ## Максимальное количество итераций градиентного спуска
    _lambda: float = 10 ** -3  ## Сила регуляризации (штрафной коэффициент)
    fit_interception: bool = True  ## Добавление константы
    verbose: bool = True
    beta: np.ndarray


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

    def fit(self, X, y):
        '''
        Обучение
        '''
        if self.fit_intercept:
            # то добавляем константу, т.е. добавляем
            # первый столбец из единиц
            X = np.c_[np.ones(X.shape[0]), X]

        # инициализация функции потерь и весов признаков
        l_prev = np.inf
        self.beta = np.zeros(X.shape[1])

        # реализация градиентного спуска
        for i in range(int(self.max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss = self._lfl(X, y, y_pred)
            if l_prev - loss < self.tol:
                return
            l_prev = loss
            self.beta -= self.lr * self._vector_gradient(X, y, y_pred)
            if self.verbose:
                info_str = f'Iteration {i}, loss {round(l_prev, 3)}, beta {np.round(self.beta, 3)}'
                print(info_str)
                logging.info(info_str)

    def _lfl(self, X, y, y_pred):
        '''
        логистическая функция потерь
        '''

        nll = -np.log(np.where(y == 1, y_pred, 1 - y_pred)).sum()
        # сильно оптимизированный код
        param = 2 if self.penalty == 'l2' else 1  # проверка на значения проведена при инициализации
        local_beta = self.beta[1:] if self.fit_intercept else self.beta
        penalty = (self._lambda / param) * np.linalg.norm(local_beta, ord=param) ** param

        return (penalty + nll) / X.shape[0]

    def _vector_gradient(self, X, y, y_pred):
        '''
        вектор градиента
        '''

        if self.penalty == 'l2':
            if self.fit_intercept:
                d_penalty = self._lambda * self.beta[1:]
                d_penalty = np.r_[self.beta[0], d_penalty]
            else:
                d_penalty = self._lambda * self.beta

        else:  # self.penalty == 'l1':
            if self.fit_intercept:
                d_penalty = self._lambda * np.sign(self.beta[1:])
                d_penalty = np.r_[self.beta[0], d_penalty]
            else:
                d_penalty = self._lambda * np.sign(self.beta)

        return -(np.dot(y - y_pred, X) + d_penalty) / X.shape[0]

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
