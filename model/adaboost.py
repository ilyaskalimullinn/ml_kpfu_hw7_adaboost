import numpy as np


class AdaBoost:

    def __init__(self, M):
        self.M = M

        self.input_weights: np.ndarray
        self.classifier_weights: np.ndarray

    def __init_weights(self, N):
        """ initialisation of input variables weights"""
        self.classifier_weights = np.ones(shape=(N,)) / N

    def update_weights(self, gt, predict, weights, weight_weak_classifiers):
        """ update weights functions DO NOT use loops"""
        pass

    def calculate_error(self, gt, predict, weights):
        """ weak classifier error calculation DO NOT use loops"""
        pass

    def calculate_classifier_weight(self, gt, predict, weights):
        """ weak classifier weight calculation DO NOT use loops"""
        pass

    def train(self, inputs, targets):
        """ train model"""

    def get_prediction(self, vectors):
        """ adaboost get prediction """
        pass
