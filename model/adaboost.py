from typing import Union

import numpy as np


class AdaBoost:

    def __init__(self, M):
        self.classifiers = []
        self.M = M
        self.classifier_weights = None

    def _init_input_weights(self, N):
        """ initialisation of input variables weights"""
        return np.ones(shape=(N,)) / N

    def update_input_weights(self, targets, predictions, input_weights, weak_classifier_weight):
        """ update weights functions DO NOT use loops"""
        a = np.exp((targets != predictions).astype(int) * weak_classifier_weight)
        input_weights = input_weights * a
        input_weights = input_weights / input_weights.sum()
        return input_weights

    def calculate_error(self, targets, predictions, weights):
        """ weak classifier error calculation DO NOT use loops"""
        return weights[targets != predictions].sum()

    def calculate_classifier_weight(self, error):
        """ weak classifier weight calculation DO NOT use loops"""
        return np.log((1 - error) / error)

    def train(self, inputs, targets):
        """ train model"""
        N = inputs.shape[0]
        input_weights = self._init_input_weights(N)
        classifier_weights = []

        for i in range(self.M):
            self.classifiers.append(self.train_weak_classifier(inputs, targets, input_weights))
            predictions = self.classifiers[i].get_predictions(inputs)
            error = self.calculate_error(targets, predictions, input_weights)
            classifier_weights.append(self.calculate_classifier_weight(error))
            input_weights = self.update_input_weights(targets, predictions, input_weights, classifier_weights[i])

            if np.isclose(error, 0.5):
                break

        self.classifier_weights = np.array(classifier_weights).reshape(1, -1)

    def get_predictions(self, inputs):
        """ adaboost get prediction """
        weak_classifier_answers = np.array([c.get_predictions(inputs) for c in self.classifiers])
        return np.sign(self.classifier_weights @ weak_classifier_answers)

    def train_weak_classifier(self, inputs, targets, input_weights):
        """ train weak classifier, adaboost decision tree """
        classifier = AdaBoostDT()
        classifier.train(inputs, targets, input_weights)
        return classifier


class AdaBoostDT:
    def __init__(self, min_metric=1e-4, min_elem=3, positive_class=1, negative_class=-1):
        self.__all_dim = None
        self.D = None
        self.min_metric = min_metric
        self.min_elem = min_elem
        self.positive_class = positive_class
        self.negative_class = negative_class

        self.max_depth = 1
        self.root = Node()

    def train(self, inputs, targets, weights):
        metric = self._calc_metrics(targets, weights)
        self.D = inputs.shape[1]
        self.__all_dim = np.arange(self.D)

        self.__get_axis, self.__get_threshold = self.__get_all_axis, self.__generate_all_threshold

        self.__build_tree(inputs, targets, weights, self.root, 0, metric)

    def __build_tree(self, inputs, targets, weights, node, depth, metric):
        # todo

        N = weights.sum()
        if depth >= self.max_depth or metric <= self.min_metric or targets.shape[0] <= self.min_elem:
            node.terminal_node = self._create_term_value(targets, weights)
        else:

            ax_max, th_max, ind_left_max, ind_right_max, disp_left_max, disp_right_max = self.__build_splitting_node(
                inputs, targets, weights, metric, N)
            node.split_ind = ax_max
            node.split_val = th_max
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[ind_left_max], targets[ind_left_max], weights[ind_left_max], node.left_child,
                              depth + 1, disp_left_max)
            self.__build_tree(inputs[ind_right_max], targets[ind_right_max], weights[ind_right_max], node.right_child,
                              depth + 1, disp_right_max)

    def __build_splitting_node(self, inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray, metric: float,
                               N: float):
        # todo
        """

        :param inputs: train inputs that came to this node
        :param targets: train targets that came to this node
        :param metric: metric (entropy or variance) for this node
        :return: feature index for feature selection function,
                threshold for splitting function,
                indexes for elements that go to the left child node,
                indexes for elements that go to the right child node,
                metric value for left node,
                metric value for right node
        """
        if N is None:
            N = weights.sum()

        information_gain_max = 0
        idx_right_best = None
        idx_left_best = None
        metric_left_best = None
        metric_right_best = None
        ax_best = None
        th_best = None

        for ax in self.__get_axis():
            for th in self.__get_threshold(inputs[:, ax]):
                idx_right = np.where(inputs[:, ax] > th, True, False)
                idx_left = ~idx_right

                information_gain, metric_left, metric_right = self.__inf_gain(targets[idx_left], targets[idx_right],
                                                                              weights[idx_left], weights[idx_right],
                                                                              N=N)

                if information_gain >= information_gain_max:
                    ax_best = ax
                    th_best = th
                    information_gain_max = information_gain
                    idx_right_best = idx_right
                    idx_left_best = idx_left
                    metric_left_best = metric_left
                    metric_right_best = metric_right

        return ax_best, th_best, idx_left_best, idx_right_best, metric_left_best, metric_right_best,

    def __inf_gain(self, targets_left: np.ndarray, targets_right: np.ndarray, weights_left: np.ndarray,
                   weights_right: np.ndarray, parent_metric: Union[float, None] = None,
                   N: Union[int, None] = None):

        if parent_metric is None:
            parent_metric = self._calc_metrics(np.hstack([targets_left, targets_right]),
                                               np.hstack([weights_left, weights_right]))
        if N is None:
            N = weights_left.sum() + weights_right.sum()

        metric_left = self._calc_metrics(targets_left, weights_left)
        metric_right = self._calc_metrics(targets_right, weights_right)

        expected_metric = (weights_left.sum() / N) * metric_left + (weights_right.sum() / N) * metric_right

        inf_gain = parent_metric - expected_metric

        return inf_gain, metric_left, metric_right

    def _create_term_value(self, targets, weights):
        positive_mask = targets == self.positive_class
        p = np.array([weights[~positive_mask].sum(), weights[positive_mask].sum()])
        if np.allclose(p, [0, 0]):
            return np.array([0.5, 0.5])
        p = p / p.sum()
        return p

    def _calc_metrics(self, targets: np.ndarray, weights: np.ndarray) -> float:
        return self._shannon_entropy(targets, weights)

    def _shannon_entropy(self, targets, weights) -> float:
        """
        :param targets: train targets that made it to current node
        :param weights: train set weights that made it to current node
        :return: entropy
        """
        p = self._create_term_value(targets, weights)
        if np.isclose(np.abs(p[0] - p[1]), 1):
            return 0
        res = -np.sum(p * np.log2(p))
        return res

    def __get_all_axis(self):
        """
        Feature selection function
        :return: all indexes of input array - a range 0...d-1
        """
        return self.__all_dim

    def __generate_all_threshold(self, inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: all train inputs (just one feature from all of it, array of shape (N, 1))
        :return: all thresholds - all unique values of this feature
        """
        return np.unique(inputs)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: вектора характеристик
        :return: предсказания целевых значений
        """
        results = []
        for input in inputs:
            node = self.root

            while node.terminal_node is None:
                if input[node.split_ind] > node.split_val:
                    node = node.right_child
                else:
                    node = node.left_child

            results.append(node.terminal_node)

        return np.vstack(results).argmax(axis=1) * 2 - 1


class Node:
    def __init__(self):
        self.right_child = None
        self.left_child = None
        self.split_ind = None
        self.split_val = None
        self.terminal_node = None
