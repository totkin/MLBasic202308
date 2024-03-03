import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000, regularization_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        reg_term = (self.regularization_param / (2 * m)) * np.sum(np.square(theta[1:]))
        cost = (1 / m) * np.sum((-y * np.log(h)) - ((1 - y) * np.log(1 - h))) + reg_term
        return cost

    def gradient_descent(self, X, y, theta):
        m = len(y)
        for _ in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, theta))
            reg_term = (self.regularization_param / m) * theta
            reg_term[0] = 0  # Don't regularize the bias term
            gradient = (1 / m) * np.dot(X.T, (h - y)) + reg_term
            theta -= self.learning_rate * gradient
        return theta

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        y = y[:, np.newaxis]  # convert to column vector
        n = X.shape[1]  # number of features
        initial_theta = np.zeros((n, 1))
        self.theta = self.gradient_descent(X, y, initial_theta)


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        return np.round(self.sigmoid(np.dot(X, self.theta)))

    def save_loss(self, loss_value):
        self.loss_history.append(loss_value)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y)) + (
                        self.regularization_param / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if p >= 0.5 else 0 for p in predictions]

    def probe(self):
        print("Current Weights:", self.weights)
        print("Current Bias:", self.bias)
        print("Regularization Parameter (L2):", self.regularization_param)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def train(self, X, y):
        # Your training code here
        # During or after each iteration, append the loss to the loss_history
        for epoch in range(self.epochs):
            # ... (training steps)
            current_loss = 1# compute and store the current loss
            self.loss_history.append(current_loss)

    def save_loss_history(self, filename="loss_history.txt"):
        with open(filename, "w") as file:
            for loss in self.loss_history:
                file.write(str(loss) + "\n")
    '''
    Эпоха (epoch) в контексте логистической регрессии с градиентным спуском
    означает один полный проход всего обучающего набора данных через алгоритм обучения.
    '''

    '''
    Процесс градиентного спуска используется для минимизации функции затрат путем поиска оптимальных параметров модели.
    В контексте машинного обучения это означает нахождение минимума функции ошибки (или потерь) путем поиска 
    оптимальных весов модели. Процесс начинается с инициализации весов модели случайными значениями.
    Затем для каждого набора обучающих данных вычисляется градиент функции ошибки по отношению к каждому весу.
    Градиент указывает направление наискорейшего роста функции, таким образом, изменение весов в направлении,
    противоположном градиенту, может уменьшить значение функции ошибки. Этот процесс повторяется до тех пор, пока
    градиент не станет близок к нулю или до тех пор, пока мы не достигнем установленного числа итераций. В результате
    модель с наименьшей ошибкой обучения определяет оптимальные параметры для задачи, на которой она обучается.
    '''