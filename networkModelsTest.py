import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Carregar seus dados pré-processados (resultado da FASE 1)
X, y = np.load('./dados_preprocessados.npy')
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir uma lista de modelos a serem testados
model_configs = [
    {'layers': [Dense(64, activation='relu', input_dim=X_train.shape[1]), Dense(1, activation='sigmoid')],
     'learning_rate': 0.001,
     'momentum': 0.9},
    # Adicione mais configurações de modelos aqui
]

# Loop para treinar e salvar os modelos
for i, config in enumerate(model_configs):
    model = Sequential(config['layers'])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'], momentum=config['momentum'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    num_trainings = 20  # Defina o número de treinamentos para cada modelo
    for j in range(num_trainings):
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # Treine o modelo

        # Salve o modelo após cada treinamento, renomeando para identificar o modelo e treinamento
        model.save(f'modelo_{i}_treinamento_{j}.h5')
