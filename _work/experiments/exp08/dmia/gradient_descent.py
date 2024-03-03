import numpy as np


def gradient_descent(x, y, theta, alpha, num_iters):
    m = theta.shape[0]
    cost_history = []
    for i in range(num_iters):
        hypothesis = theta[0].dot(x)
        cost = 1/m * np.power(hypothesis - y, 2)
        if i == 0:
            cost_history.append(cost)

        theta = theta - alpha * hypothesis.T.dot(x.T).dot(hypothesis.reshape(1,-1) - y)

        hypothesis = theta[0].dot(x)
        cost_history.append(1/m * np.power(hypothesis - y,2))

    return theta, cost_history

x = np.array([[1, 1], [1, -1], [-1, 1]])
y = np.array([1, 0, -1])
theta_initial = np.ones(x.shape[1])
alpha = 0.001
num_iters = 1000
theta, cost_history = gradient_descent(x, y, theta_initial, alpha, num_iters)
print(theta)
print('Final cost:', cost_history[-1])




def gradient_check(x, gradient_func, epsilon=1e-4):
    gradients_approx = np.zeros_like(x)
    gradients_exact = gradient_func(x)

    for i in range(len(x)):
        x_plus = np.copy(x)
        x_plus[i] += epsilon
        x_minus = np.copy(x)
        x_minus[i] -= epsilon

        gradient_approx = (gradient_func(x_plus) - gradient_func(x_minus)) / (2 * epsilon)
        gradients_approx[i] = gradient_approx

    difference = np.linalg.norm(gradients_approx - gradients_exact)
    return difference


# Пример использования

def example_function(x):
    return x ** 2


def example_gradient(x):
    return 2 * x


x = np.array([3.0, 4.0, 5.0])
difference = gradient_check(x, example_gradient)
print("Разница между аналитическим и численным градиентом:", difference)

'''В этом примере функция gradient_check принимает входное значение x, функцию gradient_func, которая вычисляет аналитический градиент, и необязательный аргумент epsilon, который определяет шаг для вычисления численного градиента.
Внутри функции мы сначала вычисляем аналитический градиент с помощью gradient_func.
Затем мы вычисляем численный градиент, используя формулу (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon), где f(x) - функция, для которой мы хотим проверить градиентный спуск.
Мы сравниваем аналитический и численный градиенты, вычисляя норму их разности с помощью np.linalg.norm.
Разница между градиентами позволяет нам оценить правильность реализации градиентного спуска. Если разница близка к нулю, значит реализация верна.'''

import plotly.graph_objects as go

# Создаем обучающие данные
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# Создаем экземпляр модели и обучаем ее
model = LogisticRegression_01()
model.fit(X, y)
probabilities = model.probe(X)
print(probabilities)

# Создаем массив для хранения номеров итераций
iteration_nums = np.arange(model.num_iterations)

# Создаем график процесса обучения
fig = go.Figure(data=go.Scatter(x=iteration_nums, y=model.loss_history))
fig.update_layout(title='Process of Model Training',
                  xaxis_title='Iteration Number',
                  yaxis_title='Loss')
fig.show()