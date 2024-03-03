import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .rlr import RegLogReg


class TestRegLogReg(unittest.TestCase):
    def setUp(self):
        '''
        Создаем экземпляр класса для тестирования всех методов
        '''
        self.model = RegLogReg()
        # self.model.precision = 3  # точность 3 знака после запятой
        n_samples = np.random.randint(1000, 5000)  # случайное количество выборок
        n_features = np.random.randint(3, 10) * 2  # количество фичей
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_classes=2, random_state=1, weights=[0.5, 0.5])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=.3, random_state=202402)

        print(f'Test X_train.shape: {self.X_train.shape}')
        self.model.verbose = False  # отключаем логирование и выводы print
        self.model.fit(self.X_train, self.y_train)
        print(f'Test fact iterations: {self.model.fact_iterations}')

    def test_fit_model(self):
        '''
        Проверяем наличие обучения модели
        '''
        self.model.verbose = False  # отключаем логирование и выводы print
        print(f'Test loss history last 3: {self.model.loss_history[-3:]}')
        self.assertIsNotNone(self.model.loss_history)  # Проверяем, что модель была обучена
        predict = self.model.predict(self.X_test)
        result = np.round(roc_auc_score(self.y_test, predict), self.model.precision)
        self.assertIsNotNone(result)
        str_message = f'Test roc_auc_score: {result}'
        print(str_message)

    def test_predict(self):
        '''
        Проверяем корректность предсказания
        '''
        self.model.verbose = False  # отключаем логирование и выводы print
        result = self.model.predict(self.X_test)  # Передаем данные для предсказания
        self.assertIsNotNone(result)  # Проверяем, что был получен результат предсказания
        print(f'Test predict is not NULL. shape={result.shape}')

    def test_probe(self):
        predictions = self.model.probe(self.X_test)
        accuracy = np.mean(predictions.round() == self.y_test)
        print(f"Accuracy {np.round(accuracy, self.model.precision)} >=0.7")
        self.assertGreaterEqual(accuracy, 0.7, "Accuracy should be greater than or equal to 70%")


if __name__ == '__main__':
    unittest.main()
