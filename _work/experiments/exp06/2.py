import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, reg_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.reg_param / m) * self.weights
            db = (1 / m) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def probe(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def gradient_check(self, X, y, epsilon=1e-5):
        approx_gradients = []
        for i in range(len(self.weights)):
            # Calculate the cost function for weights[i] + epsilon
            self.weights[i] += epsilon
            cost_plus = self._cost_function(X, y)

            # Calculate the cost function for weights[i] - epsilon
            self.weights[i] -= 2 * epsilon
            cost_minus = self._cost_function(X, y)

            # Reset the weight to original value
            self.weights[i] += epsilon

            # Calculate the approximation gradient
            approx_gradient = (cost_plus - cost_minus) / (2 * epsilon)
            approx_gradients.append(approx_gradient)

        # Calculate the actual gradients using backpropagation
        self.fit(X, y)
        actual_gradients = (1 / len(y)) * np.dot(X.T, (self.predict(X) - y))

        # Compare the actual and approximate gradients
        print(f"Approximate gradients: {approx_gradients}, Actual gradients: {actual_gradients}")

'''В методе gradient_check производится проверка, сначала считаются аппроксимированные градиенты по каждому из весов, 
затем с помощью обратного распространения ошибки вычисляются реальные градиенты. После этого происходит сравнение 
аппроксимированных и реальных градиентов.'''


import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Здесь вставьте код класса LogisticRegression

# Загрузка данных
# Пример использования датасета iris
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение модели
model = LogisticRegression(learning_rate=0.1, num_iterations=1000, reg_param=0.1)
losses = []
for i in range(1000):
    model.fit(X_train, y_train)
    loss = model._cost_function(X_train, y_train)
    losses.append(loss)

# Построение графика процесса обучения
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, 1001), y=losses, mode='lines', name='Loss'))
fig.update_layout(title='Training Loss over Iterations', xaxis_title='Iterations', yaxis_title='Loss')
fig.show()