#!/usr/bin/env python

"""ensemble.py: Implementation of a Weighted Ensemble Classifier.

The implementation follows the techniques described in
"Mining Concept-Drifting Data Streams using Ensemble Classifiers", by
Haixun Wang, Wei Fan, Philip S. Yu, Jiawei Han
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import operator
import copy as cp
import sortedcontainers as sc
from sklearn.model_selection import StratifiedKFold


__author__ = "Armita KHAJEHNASSIRI, Soumeya KAADA, Minh-Huong LE-NGUYEN"


class WeightedEnsembleClassifier:
    """
    The ensemble classifier that follows the techniques described in
    "Mining Concept-Drifting Data Streams using Ensemble Classifiers",
    by Haixun Wang, Wei Fan, Philip S. Yu, Jiawei Han.

    Attributes:
        - K: maximum number of classifiers in the ensemble
        - S: chunk size
        - base learner: the base estimator
        - CV: number of folds to compute the score of a newly added classifier
    """

    class WeightedClassifier:
        """
        An inner class that eases the control of weights and additional information
        of a base learner in the ensemble
        """

        def __init__(self, clf, weight, chunk_labels):
            """
            Create a new weighted classifier
            :param clf: the already trained classifier
            :param weight: the weight associated to it
            :param chunk_labels: the unique labels of the data chunk clf is trained on
            """
            self.clf = clf  # the model
            self.weight = weight  # the weight associated to this classifier
            self.chunk_labels = chunk_labels  # the unique labels of the data chunk the classifier is trained on

        def __lt__(self, other):
            """
            Compares an object of this class to the other by means of the weight.
            This method helps to sort the classifier correctly in the sorted list

            :param other: the other object of this class
            :return: true i.e. this object is smaller than the other object by means of their weight
            """
            return self.weight < other.weight

    def __init__(self, K=10, base_learner=DecisionTreeClassifier(), S=200, cv=5):
        """
        Create a new ensemble
        :param K: the maximum number of models allowed in this ensemble
        :param base_learner: the base learner, other classifiers will be a deep-copy of the base learner
        :param S: the chunk size
        :param cv: the number of folds for cross-validation
        """

        # top K classifiers
        self.K = K

        # base learner
        self.base_learner = base_learner

        # a sorted list if classifiers
        self.models = sc.SortedList()

        # cross validation fold
        self.cv = cv

        # chunk-related information
        self.S = S  # chunk size
        self.p = -1  # chunk pointer
        self.X_chunk = None
        self.y_chunk = None

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """
        Updates the ensemble when a new data chunk arrives (Algorithm 1 in the paper).
        The update is only launches when the chunk is filled up.

        :param X: the training data
        :param y: the training labels
        :param classes: array-like, contains all possible labels, derived from y if not provided
        :param weight: array-like, instance weight, uniform weights are assumed if not provided
        :return: self
        """
        N, D = X.shape

        # initializes everything when the ensemble is first called
        if self.p == -1:
            self.X_chunk = np.zeros((self.S, D))
            self.y_chunk = np.zeros(self.S)
            self.p = 0

        # fill up the data chunk
        for i, x in enumerate(X):
            self.X_chunk[self.p] = X[i]
            self.y_chunk[self.p] = y[i]
            self.p += 1

            if self.p == self.S:
                # reset the pointer
                self.p = 0

                # retrieve the classes and class count
                if classes is None:
                    classes, class_count = np.unique(self.y_chunk, return_counts=True)
                else:
                    _, class_count = np.unique(self.y_chunk, return_counts=True)

                # (1) train classifier C' from X by creating a deep copy from the base learner
                C_new = cp.deepcopy(self.base_learner)
                try:
                    C_new.fit(self.X_chunk, self.y_chunk)
                except NotImplementedError:
                    C_new.partial_fit(self.X_chunk, self.y_chunk, classes, weight)

                # compute the baseline error rate given by a random classifier
                baseline_score = self.compute_random_baseline(classes)

                # compute the weight of C', may do cross-validation if cv is not None
                clf_new = self.WeightedClassifier(clf=C_new, weight=0, chunk_labels=classes)
                clf_new.weight = self.compute_weight(model=clf_new, random_score=baseline_score, cv=self.cv)

                # (4) update the weights of each classifier in the ensemble, not using cross-validation
                for model in self.models:
                    model.weights = self.compute_weight(model=model, random_score=baseline_score, cv=None)

                # (5) C <- top K weighted classifiers in C U { C' }
                if len(self.models) < self.K:
                    self.models.add(value=clf_new)
                else:
                    if clf_new.weight > 0 and clf_new.weight > self.models[0].weight:
                        self.models.pop(0)
                        self.models.add(value=clf_new)

                # instance-based pruning only happens with Cost Sensitive extension
                self.do_instance_pruning()

        return self

    def do_instance_pruning(self):
        # do nothing here, will be overrode by derived classes
        pass

    def predict(self, X):
        """
        Predicts the labels of X in a general classification setting.
        The prediction is done via normalized weighted voting (choosing the maximum)

        :param X: the unseen data to give predictions
        :return: a list of shape (n-samples,) containing predictions
        """
        N, D = X.shape

        # List with size X.shape[0] and each value is a dict too,
        # Ex: [{0:0.2, 1:0.7}, {1:0.3, 2:0.5}]
        list_label_instance = []

        # use sum_weights for normalization
        sum_weights = np.sum([clf.weight for clf in self.models])

        # For each classifier in self.models, predict the labels for X
        for model in self.models:
            clf = model.clf
            pred = clf.predict(X)
            weight = model.weight
            for i, label in enumerate(pred.tolist()):
                if i == len(list_label_instance):  # maintain the dictionary
                    list_label_instance.append({label: weight / sum_weights})
                else:
                    try:
                        list_label_instance[i][label] += weight / sum_weights
                    except KeyError:
                        list_label_instance[i][label] = weight / sum_weights

        predict_weighted_voting = np.zeros(N)
        for i, dic in enumerate(list_label_instance):
            # return the key of max value in a dict
            max_value = max(dic.items(), key=operator.itemgetter(1))[0]
            predict_weighted_voting[i] = max_value

        return predict_weighted_voting

    def compute_score(self, model, X, y):
        """
        Compute the mean square error of a classifier, via the predicted probabilities.

        This code needs to take into account the fact that a classifier C trained on a
        previous data chunk may not have seen all the labels that appear in a new chunk
        (e.g. C is trained with only labels [1, 2] but the new chunk contains labels [1, 2, 3, 4, 5]

        :param model: the learner
        :param X: data of the new chunk
        :param y: labels of the new chunk
        :return: the mean square error MSE_i
        """
        N = len(y)
        labels = model.chunk_labels
        probabs = model.clf.predict_proba(X)
        sum_error = 0
        for i, c in enumerate(y):
            # if the label in y is unseen when training, skip it, don't include it in the error
            if c in labels:
                index_label_c = np.where(labels == c)[0][0]  # find the index of this label c in probabs[i]
                probab_ic = probabs[i][index_label_c]
                sum_error += (1 - probab_ic) ** 2
            else:
                sum_error += 1

        return sum_error / N

    def compute_weight(self, model, random_score, cv=None):
        """
        Compute the weight of a classifier given the random score (calculated on a random learner).
        The weight relies on either (1) MSE if it is a normal classifier,
        or (2) benefit if it is a cost-sensitive classifier

        :param model: the learner to compute w on
        :param random_score: the baseline calculated on a random learner
        :param cv: do cross validation or not, if cv is not None (and is a number) we will do cv-fold
        :return: the weight of clf
        """
        if cv is not None and type(cv) is int:
            # we create a copy because we don't want to "modify" an already trained model
            copy_model = cp.deepcopy(model)
            score = 0
            sf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
            for train_idx, test_idx in sf.split(X=self.X_chunk, y=self.y_chunk):
                X_train, y_train = self.X_chunk[train_idx], self.y_chunk[train_idx]
                X_test, y_test = self.X_chunk[test_idx], self.y_chunk[test_idx]
                try:
                    copy_model.clf.fit(X_train, y_train)
                except NotImplementedError:
                    copy_model.clf.partial_fit(X_train, y_train, copy_model.chunk_labels, None)

                score += self.compute_score(model=copy_model, X=X_test, y=y_test) / self.cv
        else:
            # compute the score on the entire data
            score = self.compute_score(X=self.X_chunk, y=self.y_chunk, model=model)

        # w = MSE_r = MSE_i
        return random_score - score

    def compute_random_baseline(self, classes):
        """
        This method computes the score produced by a random classifier, served as a baseline.
        The random score is MSE_r in case of a normal classifier, b_r in case of a cost-sensitive one

        :param classes: the unique values of label classes
        :return: the random baseline score
        """

        # if we assume uniform distribution
        # L = len(np.unique(classes))
        # MSE_r = L * (1 / L) * (1 - 1 / L) ** 2

        # if we base on the class distribution of the data --> count the number of labels
        _, class_count = np.unique(classes, return_counts=True)
        class_dist = [class_count[i] / self.S for i, c in enumerate(classes)]
        mse_r = np.sum([class_dist[i] * ((1 - class_dist[i]) ** 2) for i, c in enumerate(classes)])
        return mse_r
