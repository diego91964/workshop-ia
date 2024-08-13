import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.array([
    [2500, 55.0],
    [2700, 57.5],
    [3000, 60.0],
    [3500, 62.0],
    [3400, 61.0],
    [3200, 59.0]
])
y = np.array([85, 92, 105, 130, 120, 110])

# Criando o modelo de regressão linear
model = LinearRegression().fit(X, y)

# Coeficientes
print(f'Coeficientes: {model.coef_}')
print(f'Intercepto: {model.intercept_}')

# Previsão para um ano hipotético (e.g., Fertilizante: R$3.300, Produtividade: 60.5 sacas/hectare)
new_data = np.array([[3300, 60.5]])
predicted_price = model.predict(new_data)
print(f'Preço previsto da saca de soja: R${predicted_price[0]:.2f}')
