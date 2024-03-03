import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv')

X = data.drop('Species', axis=1)
y = data['Species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)

plt.figure()
plt.plot(logreg.coef_[0], label='Intercept')
plt.yticks([0, .1, .2, .3])
plt.grid(True)
plt.legend()

print('Coefficients: \n', logreg.coef_)

plt.show()