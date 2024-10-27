import numpy as np

# Función de pérdida de entropía cruzada binaria
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Restringir los valores dentro de un límite superior e inferior
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Derivada de la función de pérdida de entropía cruzada binaria
def binary_crossentropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))
