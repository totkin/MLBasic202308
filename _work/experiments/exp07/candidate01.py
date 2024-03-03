import numpy as np


class LogisticRegressionWithRegularization:
    def __init__(self, learning_rate=0.01, num_iterations=1000, reg_param=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _cost_function(self, X, y):
        m = X.shape[0]
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        cost = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        reg_term = self.reg_param / (2 * m) * np.sum(self.weights ** 2)  # regularization term
        return cost + reg_term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.num_iterations):
            h = self._sigmoid(np.dot(X, self.weights) + self.bias)
            dw = 1 / n_samples * np.dot(X.T, (h - y)) + (
                        self.reg_param / n_samples) * self.weights  # gradient with regularization
            db = 1 / n_samples * np.sum(h - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_predicted = [1 if i > 0.5 else 0 for i in h]
        return y_predicted

    def probe(self, X):
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return h

    def _calculate_numerical_gradients(self, X, y, epsilon=1e-5):
        grads = {}
        for i in range(len(self.weights)):
            original_weight = self.weights[i]
            self.weights[i] = original_weight + epsilon
            loss_plus_epsilon = self._cost_function(X, y)
            self.weights[i] = original_weight - epsilon
            loss_minus_epsilon = self._cost_function(X, y)
            self.weights[i] = original_weight
            gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
            grads['dw' + str(i)] = gradient
        return grads

    def gradient_check(self, X, y, epsilon=1e-5):
        self.fit(X, y)
        grads = self._calculate_numerical_gradients(X, y, epsilon)
        dw = [1 / n_samples * np.dot(X.T, (self._sigmoid(np.dot(X, self.weights) + self.bias) - y)) + (
                    self.reg_param / n_samples) * self.weights]
        for i in range(len(self.weights)):
            error = abs(grads['dw' + str(i)] - dw[i])
            print(f'Gradient check for weight {i}: {error}')
