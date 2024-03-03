import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# загрузка данных (замените 'data.csv' на путь к вашему файлу данных)
data = pd.read_csv('data.csv')

# Проверка первых 5 строк данных
data.head()

# Информация о данных (типы данных, отсутствующие значения и т.д.)
data.info()

# Описательная статистика
data.describe()

# Корреляционная матрица и тепловая карта
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Гистограммы распределения признаков
data.hist(bins=20, figsize=(12,10))
plt.show()

# Визуализация зависимостей между признаками
sns.pairplot(data, hue='target_class', diag_kind='hist')

# Boxplot для визуализации распределения данных
plt.figure(figsize=(10, 6))
sns.boxplot(x='target_class', y='feature_name', data=data)

# Другие методы визуализации и анализа данных, включая построение различных графиков, статистических тестов и анализа выбросов

# Например, построение распределения признаков по целевой переменной
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for idx, feat in enumerate(data.columns[:-1]):
    ax = axes[int(idx / 2), idx % 2]
    sns.histplot(data[feat][data.target_class == 0], kde=True, color='b', ax=ax, label='Class 0', bins=30)
    sns.histplot(data[feat][data.target_class == 1], kde=True, color='r', ax=ax, label='Class 1', bins=30)
    ax.set_title(f'Feature: {feat}')
    ax.legend()
plt.tight_layout()