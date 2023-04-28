import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return ((predictions - targets) ** 2).mean()


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return (predictions == targets).mean()


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, number_classes: int) -> np.ndarray:
    matrix = np.zeros(shape=(number_classes, number_classes))
    for t, p in zip(targets, predictions):
        matrix[t][p] += 1
    return matrix
