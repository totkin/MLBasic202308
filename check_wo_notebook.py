import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from dmia.logreg import RegLogReg

import logging

strFileName = 'RegLogReg_01'
logging.basicConfig(level=logging.INFO,
                    filename=f"logs/{strFileName}.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")


######################################
#  Загрузка данных и простейшее EDA  #
######################################

data = pd.read_csv('data/bankloan.csv', sep=';', decimal=',', engine='pyarrow')
shp = data.shape
print(f'Data shape: {shp}', f'\nColumns: {data.columns.tolist()}')
print(data.info())
print('\nData sample:\n', data.iloc[np.random.choice(shp[0], replace=False, size=5)])
logging.info(f'Data shape: {shp}')
strMessage = f'Job values: {data["job"].unique()}'
print(strMessage)
logging.info(strMessage)


###############################
#   Сплит и стандартизация    #
###############################


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

model = RegLogReg()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
result = np.round(roc_auc_score(y_test, proba), 3)
model.loss_visualize(strFileName + '_X_train', result)
