import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Загрузка данных (замените 'data.csv' на путь к вашему файлу данных)
data = pd.read_csv('data.csv')

# Проверка первых 5 строк данных
display(data.head())

# Описание данных
display(data.info())

# Описательная статистика
display(data.describe())

# Корреляционная матрица и тепловая карта
correlation_matrix = data.corr()
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.index.values,
    y=correlation_matrix.columns.values,
    colorscale='Viridis',
    colorbar=dict(title='Strength of Correlation', tickvals=[-1, 0, 1])
))
fig.update_layout(title='Correlation Heatmap')
fig.show()

# Визуализация распределения признаков
fig = px.histogram(data, x='feature_name', color='target_class', marginal='box',
                   barmode='group', histnorm='probability density')
fig.update_layout(title='Distribution of Feature by Target Class', xaxis_title='Feature Value',
                  yaxis_title='Density', bargap=0.05)
fig.show()

# Другие методы визуализации и анализа данных, включая построение различных графиков и анализа выбросов
# Например, построение распределения признаков по целевой переменной
fig = px.violin(data, y='feature_name', x='target_class', box=True,
                points="all", hover_data=data.columns)
fig.update_layout(title='Feature Distribution by Target Class', xaxis_title='Target Class',
                  yaxis_title='Feature Name', showlegend=False)
fig.show()
