import numpy as np


class FullBinaryMetrics:
    TP: float
    TN: float
    FP: float
    FN: float
    FPR: float
    FNR: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def __init__(self, TP, TN, FP, FN):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

        self.accuracy = (TP + TN) / (TP + TN + FP + FN)
        if TP == 0:
            self.precision = 0
            self.recall = 0
            self.f1_score = 0
        else:
            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        self.FPR = 0 if FP == 0 else FP / (FP + TN)
        self.FNR = 0 if FN == 0 else FN / (TP + FN)


def calc_binary_metrics(gt: np.array, predictions: np.array,
                        classes: iter = [0, 1]) -> FullBinaryMetrics:
    """
    :param gt: general truth, actual classes
    :param predictions: model predictions
    :param classes: list of size 2 that contains the classes of the objects (default [0, 1])
    :return: full metrics
    """

    TP = np.sum(gt[gt == classes[1]] == predictions[gt == classes[1]])
    TN = np.sum(gt[gt == classes[0]] == predictions[gt == classes[0]])
    FP = np.sum(gt[gt == classes[0]] != predictions[gt == classes[0]])
    FN = np.sum(gt[gt == classes[1]] != predictions[gt == classes[1]])

    return FullBinaryMetrics(TP, TN, FP, FN)


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, number_classes: int) -> np.ndarray:
    matrix = np.zeros(shape=(number_classes, number_classes))
    for t, p in zip(targets, predictions):
        matrix[t][p] += 1
    return matrix
