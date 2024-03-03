import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, beta=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta
        self.weights_ = None

    def fit(self, inputs, targets):
        n_samples = len(targets)
        self.weights_ = np.zeros(len(inputs[0]))
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(n_samples):
                z = self.predict(inputs[i])
                loss = self.loss(z, targets[i])
                self.update_weights(inputs[i], targets[i], loss)
                total_loss += loss
            print("Epoch {}/{}...".format(epoch + 1, self.epochs), end=' ')
        return self

    def predict(self, input_vector):
        return 1 / (1 + np.exp(-np.dot(input_vector, self.weights_)))

    def loss(self, prediction, target):
        return (prediction * np.log(prediction) + (1 - prediction) * np.log(
            1 - prediction)) * target + self.beta * np.sum(np.square(self.weights_))

    def update_weights(self, input_vector, target):
        gradients = np.array([
            np.multiply((prediction - target), input_val) + self.beta * weight
            for prediction, input_val, weight in zip(self.predict(input_vector), input_vector, self.weights_)
        ])
        self.weights_ += self.learning_rate * gradients / len(input_vector)


# Test code
# if __name__ == 'main':
#     # Generate dummy data
np.random.seed(42)
n_samples, n_features = 10, 5
weights = np.random.rand(n_features)
data = np.random.randn(n_samples, n_features)
targets = np.mod(np.sum(data * weights), 3)
logistic_regression = LogisticRegression()
logistic_regression.fit(data, targets)
