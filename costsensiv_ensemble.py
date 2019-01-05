import numpy as np
import heapq as hq
from skmultiflow.trees import HoeffdingTree
from ensemble import WeightedEnsembleClassifier


class CostSensitiveWeightedEnsembleClassifier(WeightedEnsembleClassifier):
    """
    An ensemble that focuses rather on cost-sensitive ensemble classifier
    """

    def __init__(self, K=10):
        """
        Initializes the classifier
        :param K:
        """
        super().__init__(K)

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """
        Algorithm 2

        :param X:
        :param y:
        :param classes:
        :param weight:
        :return:
        """
        # get the trained ensemble from X
        S = super().partial_fit(X, y, classes, weight)  # algorithm 1 to have K classifier

        # TODO Algorithm 2: do instance-based pruning here

        return self

    def predict(self, X):
        """
        Algorithm 3
        :param X:
        :return:
        """
        #TODO Implement Algorithm 3 here
        return np.zeros((X.shape[0]))