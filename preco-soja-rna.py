import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Opcional: Plotar a perda durante o treinamento
import matplotlib.pyplot as plt

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

# Normalizando os dados (opcional, mas recomendado para redes neurais)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Construindo o modelo de rede neural
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilando o modelo
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo com mais épocas
epochs = 200  # Aumentando o número de épocas para melhorar o ajuste
history = model.fit(X_normalized, y, epochs=epochs, verbose=2)

# Previsão para um ano hipotético (e.g., Fertilizante: R$3.300, Produtividade: 60.5 sacas/hectare)
new_data = np.array([[3300, 60.5]])
new_data_normalized = (new_data - X_mean) / X_std
predicted_price = model.predict(new_data_normalized)

print(f'Preço previsto da saca de soja: R${predicted_price[0][0]:.2f}')

print(history)


plt.plot(history.history['loss'])
plt.title('Perda durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda (MSE)')
plt.show()
