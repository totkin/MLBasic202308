from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_datafile(n_samples=None, n_features=None, filename=None):
    RANDOME_STATE = 202403
    if not n_samples:
        n_samples = np.random.randint(500, 3000) * 2  # кол-во выборок, для простоты баланса всегда четное
    if not n_features:
        n_features = np.random.randint(5, 20)  # количество фичей от 5 до 20

    # Создание набора данных для Логистической Регрессии
    # 2 класса, сбалансированные значения
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=RANDOME_STATE,
                               n_classes=2, weights=[0.5, 0.5])

    # Генерация названий для признаков
    feature_names = [f"feature_{i + 1}" for i in range(n_features)]

    # Преобразование в датафрейм pandas
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y  # Добавление столбца с целевыми значениями

    # Сохранение в файл
    if not filename:
        filename = f'classification_data_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    df.to_csv(filename, index=False)

    return df.shape, filename


if __name__ == '__main__':
    print(make_datafile())
