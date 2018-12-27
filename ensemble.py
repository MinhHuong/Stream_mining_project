import numpy as np
from skmultiflow.trees import HoeffdingTree

class WeightedEnsembleClassifier:
    """The classifier that follows the rules of Weighted Ensemble Classifier"""

    def __init__(self, K=10):
        # TODO

        # top K classifiers
        self.K = K

        # an array of weights
        self.weights = np.zeros((K,))

        # an array of classifiers, each classifier is associated a weight
        self.models = []


    def partial_fit(self, X, y=None, classes=None):
        # TODO
        # Implement the basic Algorithm 1 as described in the paper

        # train classifier C' from X
        C_new = HoeffdingTree() # for now let's suppose we only train Hoeffding Tree
        C_new.partial_fit(X, y)

        # compute the error rate of C' via cross-validation on X
        MSE_i = 0 # TODO
        MSE_r = 0 # TODO

        # derive weight w' for C' using (8) or (9)
        w = MSE_r - MSE_i

        # for each classifier Ci in C
        # (1) apply Ci on S to derive MSEi
        # (2) compute wi based on (8) or (9)
        for i, clf in enumerate(self.models):
            MSE_i = 0 # TODO
            self.weights[i] = MSE_r - MSE_i # TODO

        # C <- top K weighted classifiers in C U { C' }
        top_K_indices = self.weights.argsort()[::-1][:self.K] # descending sort of the weights
        C = self.models[top_K_indices]

        # TODO need to take into account C_new as well

        return C

    def predict(self, X):
        # TODO
        N = X.shape[0]
        return np.zeros((N,))