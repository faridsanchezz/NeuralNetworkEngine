from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def display_loss_per_epoch(losses):
    epochs = len(losses)

    # Graficar la pérdida por epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b', label='Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.show()


def confusionMatrix(y_true, y_pred):
    # Convertir las probabilidades en etiquetas
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Generar matriz de confusión
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(conf_matrix)


def accuracy(y_test, y_pred):
    # Evaluación final en el conjunto de prueba
    final_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Final Accuracy on Test Set: {final_accuracy * 100:.2f}%")