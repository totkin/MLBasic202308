import numpy as np
from abc import ABC


def sigmoid(z):
    '''
    Вычисление сигмоиды
    '''
    return 1 / (1 + np.exp(-z.astype(float)))


def loss(h, y) -> float:
    '''
    Логистическая функция потерь
    '''
    return float((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean())


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


class BaseLG(ABC):
    init_method: str  # Метод инициализации весов
    lr: float = .01  # Темп обучения
    num_iter: int = 10 ** 5  # Кол-во итераций градиентного спуска
    tol: float = 10 ** -5  # Допуск сходимости
    fit_interception: bool = True  # Добавление константы
    verbose: bool = True  # Логирование результатов оптимизации
    loss_by_iter_ = []  # Список значений функции потерь по итерациям
    theta = np.ndarray
