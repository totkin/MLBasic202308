import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, regularization_strength=0.1, num_iters=1000, num_features=1,
                 label_values=None):
        """
        Initialize the logistic regression model.

        Args:
            learning_rate (float): The learning rate.
            regularization_strength (float): Strength of the regularization.
            num_iters (int): Number of iterations.
            num_features (int): The number of features in the data.
            label_values (array): The label values.
        """

        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_iters = num_iters
        self.num_features = num_features
        self.label_values = label_values

    def fit(self, data, labels):
        """
            Fit the model on the given data and labels.
        Args:
        data (array): A 2D array containing the training data. Each row is a sample and each column is a feature.
        labels (array): An array containing the labels for the samples.
                    Returns:
                        None
                """
        # Convert data to NumPy array
        data = np.array(data)
        labels = np.array(labels)

        # Initialize weights and bias
        self.weights = np.random.randn(self.num_features + 1)
        self.bias = 0

        for _ in range(self.num_iters):
            # Compute sigmoid function for each sample
            predictions = self.sigmoid(data.dot(self.weights.reshape(-1, 1)) + self.bias)
            loss = self.compute_loss(predictions, labels)

            if self.regularization_strength > 0:
                loss += self.regularize_loss(self.weights)

                self.adjust_weights(data, loss)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def compute_loss(predictions, labels):
        total_loss = 0.0

        for prediction, label in zip(predictions, labels):
            if label == 1:
                total_loss += -np.log(prediction)
            elif label == 0:
                total_loss += -np.log(1 - prediction)

        return total_loss

    def regularize_loss(self, weights):
        regularization_term = 0.5 * self.regularization_strength * np.sum(weights ** 2)
        return regularization_term

    def adjust_weights(self, data, loss):
        gradient = self.get_gradient(data, loss)
        self.weights -= gradient * self.learning_rate

    def get_gradient(self, data, loss):
        predictions = self.sigmoid(data.dot(self.weights.reshape(-1, 1)) + self.bias)
        gradients = np.zeros(self.weights.shape)
        for i, prediction in enumerate(predictions):
            gradient_x = prediction - labels[i]
            gradients += data[:, i] * gradient_x

            if self.label_values is not None:
                gradient_y = gradient_x * self.label_values
                gradients += np.ones(len(self.label_values)) * gradient_y

        gradients /= len(data)

        return gradients


    def grad_descent(f, x, gradient, alpha):
        for _ in range(100):
            x -= gradient(x) * alpha
            if np.linalg.norm(gradient(x)) < 1e-5:
                break
        return x


    def sigmoid_cross_entropy_with_logits(y_true, y_pred):
        y_pred_exp = np.exp(y_pred)
        y_pred_softmax = y_pred_exp / y_pred_exp.sum()
        cross_entropy = -y_true.dot(np.log(y_pred_softmax))
        return cross_entropy


x = np.linspace(-3, 3, num=10)
y = np2.linspace(-2, 2, num=20)
X, Y = np.meshgrid(x, y)

# вот пример вызова и использования этого класса в Jupyter Notebook:

# импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# загружаем датасет
dataset = pd.read_csv('data.csv')

# разбиваем датасет на обучающую и тестовую выборки
X = dataset.drop('target', axis=1)  # все признаки, кроме целевого
y = dataset['target']  # целевой признак
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# создаём объект логистической регрессии
logreg = LogisticRegression(random_state=0).fit(X_train, y_train)

# делаем предсказания на тестовой выборке и считаем точность
y_pred = logreg.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=3, cols=1)

for i in range(3):
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1]), row=i + 1, col=1
    )

fig.update_layout(height=600, width=800)
fig.show()

from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Загрузка данных
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Используем только лепестки и листья
y = (iris.target == 2).astype(int)  # Целевой признак - это вид Iris versicolor

# Создание pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())

# Обучение модели
pipe.fit(X, y)

# Вычисление кривых обучения
train_sizes, train_scores, val_scores = learning_curve(pipe, X, y, cv=3)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.
