import os

import numpy as np

from datasets.dataset_titanic import Titanic
from metrics.binary_metrics import calc_binary_metrics, confusion_matrix
from model.adaboost import AdaBoost


def build_confusion_matrix(targets, predictions):
    conf_matrix_pred = np.array((predictions + 1) // 2).astype(int).reshape(-1)
    conf_matrix_targets = ((targets + 1) // 2).astype(int)
    return confusion_matrix(conf_matrix_pred, conf_matrix_targets, 2)


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.curdir)

    titanic = Titanic(os.path.join(ROOT_DIR, "static/titanic_train_data.csv"),
                      os.path.join(ROOT_DIR, "static/titanic_test_data.csv"))

    dataset = titanic()
    model = AdaBoost(30)
    model.train(dataset["train_input"], dataset["train_target"])

    predictions = model.get_predictions(dataset["test_input"])
    model_metrics = calc_binary_metrics(dataset["test_target"].reshape(-1), predictions.reshape(-1), classes=[-1, 1])

    print("Metrics on test set:")
    print(f"Accuracy: {round(model_metrics.accuracy, 2)}; precision: {round(model_metrics.precision, 2)}")
    print(f"Recall: {round(model_metrics.recall, 2)}; f1-score: {round(model_metrics.f1_score, 2)}")
    print("Confusion matrix:")
    print(build_confusion_matrix(dataset['test_target'], predictions))
