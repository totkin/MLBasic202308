import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Добавление столбца с единицами для bias
        X = np.insert(X, 0, 1, axis=1)

        # инициализация весов
        self.theta = np.zeros(X.shape[1])

        # градиентный спуск
        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * (gradient + self.regularization_param * self.theta) / y.size

    def predict(self, X):
        # Добавление столбца с единицами для bias
        X = np.insert(X, 0, 1, axis=1)
        return np.round(self.sigmoid(np.dot(X, self.theta)))


# Пример использования
# Инициализация модели
model = LogisticRegression(learning_rate=0.1, num_iterations=3000, regularization_param=0.1)

# Обучение модели
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

model.fit(X_train, y_train)

# Предсказание
X_test = np.array([[1, 3], [2, 4]])
predictions = model.predict(X_test)
print(predictions)

'''
        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m + (self.regularization_param / m) * self.theta
            self.theta -= self.learning_rate * gradient
            
Этот код представляет градиентный спуск в контексте логистической регрессии с регуляризацией на основе второй
производной. Вот пошаговый разбор:

for _ in range(self.num_iterations)
Этот цикл выполняется num_iterations раз. Он предполагает, что обучение будет повторяться несколько раз для того,
чтобы параметры модели могли сойтись к оптимальному значению.

z = np.dot(X, self.theta)
Здесь вычисляется линейная комбинация входных признаков X с использованием весов theta.

h = self.sigmoid(z)
Далее, результат z передается через функцию активации (сигмоид) для получения предсказанных значений.

gradient = np.dot(X.T, (h - y)) / m + (self.regularization_param / m) * self.theta
Градиент регуляризованной функции потерь вычисляется путем взвешивания разницы между предсказаниями и 
реальными значениями на входе X, затем добавляется регуляризационный член.

self.theta -= self.learning_rate * gradient: После чего обновляются веса theta на основе градиентного спуска
с указанным learning_rate.

Этот код в цикле выполняет итерации градиентного спуска для обучения модели логистической регрессии с регуляризацией
по стандартной формуле градиента итеративно обновляет веса модели в направлении минимизации функции потерь.


'''


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.theta = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m + (self.regularization_param / m) * self.theta
            self.theta -= self.learning_rate * gradient

            loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (self.regularization_param / (2 * m)) * np.sum(self.theta**2)
            self.loss_history.append(loss)

    def predict(self, X):

        # Добавление столбца с единицами для bias
        z = np.dot(X, self.theta)
        return self.sigmoid(z)


    def predict(self, X):
        # Добавление столбца с единицами для bias
        X = np.insert(X, 0, 1, axis=1)
        return np.round(self.sigmoid(np.dot(X, self.theta)))

    import numpy as np

    class LogisticRegression:
        def __init__(self, num_iterations=100, learning_rate=0.01, regularization_param=0.01):
            self.num_iterations = num_iterations
            self.learning_rate = learning_rate
            self.regularization_param = regularization_param
            self.loss_history = []  # To store loss at each iteration

        def fit(self, X, y):
            m, n = X.shape
            self.theta = np.zeros(n)

            for _ in range(self.num_iterations):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                loss = self.calculate_loss(h, y, self.theta, m)
                self.loss_history.append(loss)  # Save loss at each iteration

                gradient = self.calculate_gradient(X, h, y, m)
                self.theta -= self.learning_rate * gradient

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def calculate_loss(self, h, y, theta, m):
            epsilon = 1e-15
            loss = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()
            regularization_term = (self.regularization_param / (2 * m)) * np.sum(np.square(theta[1:]))
            return loss + regularization_term

        def calculate_gradient(self, X, h, y, m):
            gradient = np.dot(X.T, (h - y)) / m + (self.regularization_param / m) * self.theta
            return gradient


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            loss = -1 / n_samples * (np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)))
            self.loss_history.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return [1 if pred >= 0.5 else 0 for pred in predictions]

    def check_gradient(self, X, y, epsilon=1e-5):
        for idx in range(3):
            # Assuming 3 features for simplicity
            w_plus_epsilon = self.weights.copy()
            w_minus_epsilon = self.weights.copy()

            w_plus_epsilon[idx] += epsilon
            w_minus_epsilon[idx] -= epsilon

            loss_plus = self.compute_loss(X, y, w_plus_epsilon)
            loss_minus = self.compute_loss(X, y, w_minus_epsilon)

            gradient_approx = (loss_plus - loss_minus) / (2 * epsilon)
            gradient_computed = (1 / len(y)) * np.dot(X[:, idx], self.sigmoid(np.dot(X, self.weights) + self.bias) - y)

            # Compare gradient computed by function and manually calculated gradient
            print(f'Feature index: {idx}, Computed Gradient: {gradient_computed}, Approx Gradient: {gradient_approx}')

        return None

    def compute_loss(self, X, y, weights):
        model = np.dot(X, weights) + self.bias
        predictions = self.sigmoid(model)
        loss = -1 / len(y) * (np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)))
        return loss
