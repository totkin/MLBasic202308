import numpy as np


class RegLogReg:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # Вычисление сигмоидной функции
        return 1 / (1 + np.exp(-z))

    def _initialize_weights(self, num_features):
        # Инициализация весов случайными значениями
        self.weights = np.random.rand(num_features)
        self.bias = 0

    def fit(self, X, y):
        # Шаг 1: Инициализация весов
        num_samples, num_features = X.shape
        self._initialize_weights(num_features)

        # Шаг 2: Градиентный спуск
        for i in range(self.num_iterations):
            # Шаг 2.1: Вычисление линейной комбинации весов и признаков
            linear_model = np.dot(X, self.weights) + self.bias

            # Шаг 2.2: Применение сигмоидной функции к линейной комбинации
            y_pred = self._sigmoid(linear_model)

            # Шаг 2.3: Вычисление градиентов для весов и смещения
            dw = np.dot(X.T, (y_pred - y)) / num_samples
            db = np.sum(y_pred - y) / num_samples

            # Шаг 2.4: Регуляризация градиентов
            dw += (2 * self.regularization_param * self.weights) / num_samples

            # Шаг 2.5: Обновление весов и смещения с использованием градиентного спуска
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Шаг 2.6: Сохранение результатов итерации
            self.loss_history.append(self._calculate_loss(X, y))

    def _calculate_loss(self, X, y):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        # Вычисление функции потерь с использованием логистической функции потерь
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # Добавляем регуляризацию к функции потерь
        regularization_loss = (
                self.regularization_param / (2 * X.shape[0]) * np.sum(self.weights ** 2)
        )

        loss += regularization_loss

        return loss

    def predict_proba(self, X):
        # Предсказание вероятностей классов
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return y_pred

    def predict(self, X, threshold=0.5):
        # Предсказание меток классов
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        return y_pred
