import numpy as np
from mnist import MNIST
from sklearn import datasets


def one_hot_encode(y):
    num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]


def train_test_split(X, y, test_size=0.2, random_state=None):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data_iris():
    iris = datasets.load_iris()
    X = iris.data  # Características
    y = iris.target  # Etiquetas
    y_encoded = one_hot_encode(y)

    # Dividir datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


def load_and_preprocess_data_mnist(pathMnistData):
    mndata = MNIST(pathMnistData)
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    # Convertir a numpy arrays y normalizar los valores de píxeles
    X_train, X_test = np.array(X_train) / 255.0, np.array(X_test) / 255.0
    y_train, y_test = np.array(y_train), np.array(y_test)

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test
