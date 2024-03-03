from abc import ABC, abstractmethod
import plotly.graph_objects as go
import numpy as np

class LRBase(ABC):
    def __init__(self, regularization_param=0.1, learning_rate=0.1, iterations=1000, fit_intercept=True):
        self.theta = None
        self.regularization_param = regularization_param
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss_history = []
        self.fit_intercept = fit_intercept

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def add_intercept(self, X):
        if self.fit_intercept:
            return np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    @abstractmethod
    def _compute(self, X, y):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def plot_loss_history(self):
        iterations = list(range(1, len(self.loss_history) + 1))
        fig = go.Figure(data=go.Scatter(x=iterations, y=self.loss_history, mode='lines'))
        fig.update_layout(title='Loss History', xaxis_title='Iterations', yaxis_title='Loss')
        fig.show()

class LogisticRegression(LRBase):

    def __init__(self, learning_rate=0.01, iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.fit_intercept = fit_intercept

    def _compute_cost(self, X, y):
        m = X.shape[0]
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        cost = -1 / m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost

    def _compute_gradient(self, X, y):
        m = X.shape[0]
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        dw = 1 / m * np.dot(X.T, (predictions - y))
        db = 1 / m * np.sum(predictions - y)
        return dw, db

    def fit(self, X, y):
        X = self.add_intercept(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iterations):
            dw, db = self._compute_gradient(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = self.add_intercept(X)
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        predicted_labels = [1 if i > 0.5 else 0 for i in predictions]
        return np.array(predicted_labels)