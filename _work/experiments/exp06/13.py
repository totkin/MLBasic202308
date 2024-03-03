import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))  # Добавляем столбец с единицами для сдвига

        self.theta = np.zeros(n + 1)  # Инициализируем параметры модели

        for i in range(self.num_iterations):
            y_pred = self.sigmoid(np.dot(X, self.theta))

            grad = np.dot(X.T, (y_pred - y)) / m
            self.theta[0] -= self.learning_rate * grad[0]
            self.theta[1:] = self.theta[1:] * (
                        1 - self.learning_rate * self.lambda_param / m) - self.learning_rate * grad[1:]

    def predict(self, X):
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))  # Добавляем столбец с единицами для сдвига

        y_pred = self.sigmoid(np.dot(X, self.theta))

        return np.round(y_pred)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))