import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from _work.experiments.exp01.LogisticRegression_GD_wo_tol import LogisticRegression_GD_wo_tol

import logging
from sklearn.feature_extraction.text import TfidfVectorizer

strFileName = 'RegLogReg_02'
logging.basicConfig(level=logging.INFO,
                    filename=f"logs/{strFileName}.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")


######################################
#  Загрузка данных и простейшее EDA  #
######################################

data = pd.read_csv('../../../data/IMDB_Dataset.zip', engine='pyarrow', compression='zip')
shp = data.shape

data.rename(columns={'sentiment': 'target'}, inplace=True)
data['target'] = data['target'].map({'positive': 1, 'negative': 0})

print(f'Data shape: {shp}', f'\nColumns: {data.columns.tolist()}')
print(data.info())
print('\nData sample:\n', data.iloc[np.random.choice(shp[0], replace=False, size=5)])
logging.info(f'Data shape: {shp}')
strMessage = f'target values: {data["target"].unique()}'
print(strMessage)
logging.info(strMessage)


d=data
review_summaries=d['review'].map(str.lower).tolist()
print(review_summaries[:5])

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(review_summaries)
print(vectorizer2.get_feature_names_out()[10024:10030])
print(f'X2.shape: {X2.shape}\n')

#
# ###############################
# #   Сплит и стандартизация    #
# ###############################
#
#
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1),
    data['target'],
    test_size=.3,
    stratify=data['target'],
    random_state=202402
)
#
# num_cols = 'age debtinc creddebt othdebt'.split(' ')
# logging.info(f"num_cols = {num_cols}")
# X_train_unstandardized = X_train[num_cols]
#
# standardscaler = StandardScaler()
# standardscaler.fit(X_train[num_cols])
# X_train[num_cols] = standardscaler.transform(X_train[num_cols])
# X_test[num_cols] = standardscaler.transform(X_test[num_cols])
#
# X_train = pd.get_dummies(X_train)
# X_test = pd.get_dummies(X_test)
#
#
# ####################
# #   Моделирование  #
# ####################
#
model = LogisticRegression_GD_wo_tol()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
result = np.round(roc_auc_score(y_test, proba), 3)
model.loss_visualize(strFileName + '_X_train', result)
