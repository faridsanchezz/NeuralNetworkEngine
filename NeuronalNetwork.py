import numpy as np


# Inicialización de pesos avanzada: Xavier/Glorot initialization
def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))


# Clase para representar una capa de la red neuronal con opciones avanzadas
class NeuralLayer:
    def __init__(self, input_size, output_size, activation_function, activation_derivative, use_bias=True):
        self.weights = initialize_weights(input_size, output_size)
        self.bias = np.zeros((1, output_size)) if use_bias else None
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.input = None
        self.output = None  # resultado despues de aplicar la funcion de activacion
        self.z = None  # La salida antes de aplicar la función de activación
        self.delta = None  # El error en esta capa durante el backpropagation
        self.prev_weight_gradients = 0  # Inicialización para el momento

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights)
        if self.bias is not None:
            self.z += self.bias
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, error, optimizer, momentum=0.0):
        delta_activation = self.activation_derivative(self.z)
        self.delta = error * delta_activation
        weight_gradients = np.dot(self.input.T, self.delta) / len(self.input)

        # Momento
        weight_gradients += momentum * self.prev_weight_gradients
        self.prev_weight_gradients = weight_gradients

        # Actualización de pesos y sesgos con el optimizador
        self.weights = optimizer.update(self.weights, weight_gradients)

        if self.bias is not None:
            # Asegurar que los gradientes de sesgo tengan las dimensiones correctas
            bias_gradients = np.sum(self.delta, axis=0, keepdims=True) / len(self.input)
            self.bias = optimizer.update(self.bias, bias_gradients)

        return self.delta.dot(self.weights.T)


# Clase para representar la red neuronal completa con opciones avanzadas
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output  # salida de la ultima capa, es decir, predicción de la red

    def backward(self, error, optimizer, momentum=0.0):
        for layer in reversed(self.layers):
            error = layer.backward(error, optimizer, momentum)
