from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from LogisticRegression_GD_wo_tol import LogisticRegression_GD_wo_tol
from LogisticRegression_GD_w_tol import LogisticRegression_GD_w_tol

import logging

# strFileName=Path(__file__).name
# strFileName = 'LogisticRegression_GD_wo_tol'
strFileName = 'LogisticRegression_GD_w_tol'
logging.basicConfig(level=logging.INFO,
                    filename=f"logs/{strFileName}.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")

# %matplot inline
# %config InLineBackend.figure_format = 'retina'

data = pd.read_csv('data/bankloan.csv', sep=';', decimal=',', engine='pyarrow')
shp = data.shape
print(f'Data shape: {shp}', f'\nColumns: {data.columns.tolist()}')
print(data.info())
print('\nData sample:\n', data.iloc[np.random.choice(shp[0], replace=False, size=5)])
logging.info(f'Data shape: {shp}')
strMessage = f'Job values: {data["job"].unique()}'
print(strMessage)
logging.info(strMessage)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('default', axis=1),
    data['default'],
    test_size=.3,
    stratify=data['default'],
    random_state=202402
)

num_cols = 'age debtinc creddebt othdebt'.split(' ')
logging.info(f"num_cols = {num_cols}")
X_train_unstandardized = X_train[num_cols]

standardscaler = StandardScaler()
standardscaler.fit(X_train[num_cols])
X_train[num_cols] = standardscaler.transform(X_train[num_cols])
X_test[num_cols] = standardscaler.transform(X_test[num_cols])

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

####################
#   Моделирование  #
####################

# model = LogisticRegression_GD_wo_tol(init_method='zero', lr=.05, num_iter=10)
# model.fit(X_train_unstandardized, y_train)
# model.loss_visualize(strFileName+'_X_train_unstandardized')
# print(model.loss_by_iter_)

# model_n = LogisticRegression_GD_wo_tol(init_method='zero', lr=.05, num_iter=1000)
# model_n.fit(X_train, y_train)
# proba = model_n.predict_proba(X_test)
# result = np.round(roc_auc_score(y_test, proba), 3)
# logging.info(f"AUC-ROC on test sample: {result}")
# model_n.loss_visualize(strFileName+'_1000_X_train_1000', result)


# model_m = LogisticRegression_GD_w_tol(lr=.1, num_iter=200, verbose=True)
# model_m.fit(X_train_unstandardized, y_train)
# proba = model_m.predict_proba(X_test)
# result = np.round(roc_auc_score(y_test, proba), 3)
# logging.info(f"AUC-ROC on test sample: {result}")
# model_m.loss_visualize(strFileName+'_200_X_train_unstandardized', result)


model_m = LogisticRegression_GD_w_tol(lr=.1, num_iter=1000, verbose=True)
model_m.fit(X_train, y_train)
proba = model_m.predict_proba(X_test)
result = np.round(roc_auc_score(y_test, proba), 3)
logging.info(f"AUC-ROC on test sample: {result}")
model_m.loss_visualize(strFileName+'_1000_X_train', result)
