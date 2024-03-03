import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import sigmoid, add_intercept


class LRBase(ABC):
    fit_intercept: bool = True
    loss_history = []
    precision: int = 5
    verbose: bool = False
    filename: str = Path(__file__).name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    fact_iterations: int = 0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z.astype(float)))

    def add_intercept(self, X):
        if self.fit_intercept:
            return np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def probe(self, X):
        pass

    def plot_loss_history(self, write_to_file=True):
        if self.verbose:
            iterations = list(range(1, len(self.loss_history) + 1))
            fig = go.Figure(data=go.Scatter(x=iterations, y=self.loss_history, mode='lines'))
            fig.update_layout(title=f'Loss History. Number of iteration: {self.fact_iterations}' +
                                    f'. Final loss: {round(self.loss_history[-1], self.precision)}',
                              xaxis_title='Iterations', yaxis_title='Loss')
            if write_to_file:
                fig.write_image("assets/" + self.filename + ".png")
            else:
                fig.show()
        else:
            print(f'loss: {[round(num, self.precision) for num in self.loss_history]}')


class RegLogReg(LRBase):
    def __init__(self, regularization_param=0.1, learning_rate=0.1, iterations=10 ** 5, precision=5):
        if self.verbose:
            logging.basicConfig(level=logging.INFO,
                                filename="logs/" + self.filename + ".log",
                                filemode="w",
                                format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")

        self.theta = None
        '''
        self.theta в классе RegLogReg используется для хранения параметров модели логистической регрессии.
        Каждый элемент theta соответствует весам, используемым для предсказания целевой переменной на основе
        входных признаков.
        Обычно в классе регуляризации логистической регрессии (Logistic Regression)
        используется только один параметр регуляризации, который обычно называется
        regularization_param и используется для управления сложностью модели и предотвращения переобучения.
        Параметр learning_rate используется в процессе градиентного спуска для управления скоростью обучения.
        Обычно эти два параметра представляют разные аспекты процесса обучения:
            regularization_param контролирует сложность модели,
            в то время как learning_rate определяет скорость сходимости оптимизационного алгоритма.
        '''
        self.regularization_param = regularization_param
        self.learning_rate = learning_rate
        self.precision = precision  ## Степень допуска сходимости 5 == 10**-5
        self.iterations = iterations

    def _compute(self, X, y):
        m = X.shape[0]
        h = sigmoid(np.dot(X, self.theta))
        reg_term = (self.regularization_param / (2 * m)) * np.sum(self.theta[1:] ** 2)
        loss = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) + reg_term

        reg_term = (self.regularization_param / m) * self.theta
        reg_term[0] = 0  # Not regularizing bias term
        grad = (1 / m) * np.dot(X.T, (h - y)) + reg_term

        return loss, grad

    def fit(self, X, y):
        X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        i = 0

        if self.verbose:
            info_str = f'X.shape:{X.shape}'
            print(info_str)
            logging.info(info_str)

        for i in range(self.iterations):
            loss, gradient = self._compute(X, y)
            self.theta = self.theta - self.learning_rate * gradient
            self.loss_history.append(loss)
            if (len(self.loss_history) >= 2
                    and self.loss_history[-2] - self.loss_history[-1] < 10 ** -self.precision):
                self.fact_iterations = i
                return
            if self.verbose:
                info_str = f'Iteration {i}, loss {round(loss, self.precision)}, theta {np.round(self.theta.astype(float), self.precision)}'
                print(info_str)
                logging.info(info_str)

        self.fact_iterations = i

    def predict(self, X):
        X = self.add_intercept(X)
        return np.round(sigmoid(np.dot(X, self.theta)))

    def probe(self, X):
        X = self.add_intercept(X)
        return sigmoid(np.dot(X, self.theta))


def simple_experiment():
    '''
    Проверка на простом массиве данных
    '''
    # Создаем набор данных для проверки
    num_samples, num_features = np.random.randint(200, 1000), np.random.randint(5, 10)
    X_train = np.random.rand(num_samples, num_features)
    y_train = np.random.randint(2, size=num_samples)

    # Инициализируем/обучаем модель логистической регрессии
    model = RegLogReg()
    model.verbose = True
    model.fit(X_train, y_train)

    # Рисуем график и выводим получившиеся параметры
    model.plot_loss_history()
    print("Theta:", model.theta)


def data1_experiment():
    # Создаем набор данных для тестирования
    data = pd.read_csv('data/bankloan.csv', sep=';', decimal=',', engine='pyarrow')
    shp = data.shape
    print(f'Data shape: {shp}', f'\nColumns: {data.columns.tolist()}')
    print(data.info())
    print('\nData sample:\n', data.iloc[np.random.choice(shp[0], replace=False, size=5)])
    logging.info(f'Data shape: {shp}')

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('default', axis=1),
        data['default'],
        test_size=.3,
        stratify=data['default'],
        random_state=202402
    )

    num_cols = 'age debtinc creddebt othdebt'.split(' ')
    logging.info(f"num_cols = {num_cols}")
    X_train_unstandardized = X_train[num_cols]

    standardscaler = StandardScaler()
    standardscaler.fit(X_train[num_cols])
    X_train[num_cols] = standardscaler.transform(X_train[num_cols])
    X_test[num_cols] = standardscaler.transform(X_test[num_cols])

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Инициализируем модель логистической регрессии
    model = RegLogReg()
    model.verbose = True
    model.fit(X_train, y_train)

    # Рисуем график и выводим получившиеся параметры

    print(f'Number of iteration: {model.fact_iterations}')

    probe = model.probe(X_test)
    result = np.round(roc_auc_score(y_test, probe), model.precision)
    str_message = f'roc_auc_score: {result}'
    print(str_message)
    logging.info(str_message)

    print("Theta:", model.theta)

    model.plot_loss_history()


def data2_experiment():
    # Создаем набор данных для тестирования
    strFileName2 = 'data/train.csv'
    df2 = pd.read_csv(strFileName2, sep=',')
    df2.name = 'Amazone Food Review'
    df2.rename(columns={'Prediction': 'target', 'Reviews_Summary': 'review'}, inplace=True)
    print(f'Data shape: {df2.shape}', f'\nColumns: {df2.columns.tolist()}')
    print(df2.info())
    print('\nData sample:\n', df2.iloc[np.random.choice(df2.shape[0], replace=False, size=5)])
    logging.info(f'Data shape: {df2.shape}')

    text_transformer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=15000)
    X2 = text_transformer.fit_transform(df2['review'])
    print(type(X2))
    print(f'X2.shape: {X2.shape}\n')

    y = df2.target.values
    X_train, X_test, y_train, y_test = train_test_split(X2, y, train_size=0.7, random_state=202402)

    # Инициализируем модель логистической регрессии

    model = RegLogReg()
    print(f'Number of iteration: {model.fit(X_train, y_train)}')
    probe = model.probe(X_test)
    str_message = f'probe: {probe}'
    logging.info(str_message)
    print(str_message)
    result = np.round(roc_auc_score(y_test, probe), model.precision)
    print(f'result: {result}')

    model.plot_loss_history()

    # Выводим получившиеся параметры
    print("Theta:", model.theta)


if __name__ == '__main__':
    log_name: str = "logs/" + Path(__file__).name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_.log"
    logging.basicConfig(level=logging.INFO,
                        filename=log_name,
                        filemode="w",
                        format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")
    # simple_experiment()
    data1_experiment()
    # data2_experiment()
    # data3_experiment()
