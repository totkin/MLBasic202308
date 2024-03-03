import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Model definition
model_lr = LinearRegression()

# Fitting training set into model
model_lr.fit(X_train, y_train)

# Predicting using Linear Regression Model
Prediction = model_lr.predict(X_test)

# Evaluating variance
Variance = np.var(Prediction)  # O/P : 56.55085515632455


# Evaluating SSE
SSE = np.mean((np.mean(Prediction) - y)** 2)
# O/P : 85.03300391390214

# Evaluating Variance
Bias = SSE - Variance
# O/P : 28.482148757577583