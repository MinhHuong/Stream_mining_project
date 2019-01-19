import numpy as np
from sklearn.tree import DecisionTreeClassifier
import operator
import copy as cp
import sortedcontainers as sc


class WeightedEnsembleClassifier:
    """The classifier that follows the rules of Weighted Ensemble Classifier"""

    class WeightedClassifier:
        """
        An inner class that eases the control of weights and additional information
        of a classifier in the ensemble
        """

        def __init__(self, clf, weight, chunk_labels):
            """
            Create a new weighted classifier
            :param clf: the already trained classifier
            :param weight: the weight associated to it
            :param chunk_labels: the unique labels of the data chunk clf is trained on
            """
            self.clf = clf # the model
            self.weight = weight # the weight associated to this classifier
            self.chunk_labels = chunk_labels # the unique labels of the data chunk the classifier is trained on

        def __lt__(self, other):
            """
            Compares an object of this class to the other, by means of its weight.
            This method helps the heap to sort the classifier correctly

            :param other: the other object of this class
            :return: true i.e. this object is smaller than the other object by means of their weight
            """
            return self.weight < other.weight

    def __init__(self, K=10, base_learner=DecisionTreeClassifier(), S=200):
        """
        Create a new ensemble
        :param K:       the maximum number of models allowed in this ensemble
        :param learner: indicates the weak/individual learners in the ensemble,
                        possible choices are: "tree" for Hoeffding tree,
                        "bayes" for Naive Bayes
                        TODO have RIPPER also ? https://algorithmia.com/algorithms/weka/JRip
        """

        # top K classifiers
        self.K = K

        # base learner
        self.base_learner = base_learner

        # a sorted list if classifiers
        self.models = sc.SortedList()

        # chunk-related information
        self.S = S  # chunk size
        self.p = -1  # chunk pointer
        self.X_chunk = None
        self.y_chunk = None
        self.full_chunk = False

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """
        Fit the ensemble to a data chunk
        Implement the basic Algorithm 1 as described in the paper

        :param X: the training data (a data chunk S)
        :param y: the training labels
        :param classes: array-like, contains all possible labels,
                        if not provided, it will be derived from y
        :param weight:  array-like, instance weight
                        if not provided, uniform weights are assumed
        :return: self
        """
        N, D = X.shape

        # see if we have enough data to start training
        if self.p == -1:
            self.X_chunk = np.zeros((self.S, D))
            self.y_chunk = np.zeros(self.S)
            self.p = 0

        for i, x in enumerate(X):
            self.X_chunk[self.p] = X[i]
            self.y_chunk[self.p] = y[i]
            self.p += 1
            self.full_chunk = False

            if self.p == self.S:
                # reset the pointer
                self.p = 0
                self.full_chunk = True

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

                # MSE_r: compute the baseline error rate given by a random classifier
                baseline_score = self.compute_random_baseline(classes)

                # (3) derive weight w_new for C_new using (8) MSE or (9) benefit
                clf_new = self.WeightedClassifier(clf=C_new, weight=0, chunk_labels=classes)
                w_new = self.compute_weight(model=clf_new, random_score=baseline_score)
                clf_new.weight = w_new

                # (4) update the weights of each classifier in the ensemble
                for i, model in enumerate(self.models):
                    model.weights = self.compute_weight(model=model, random_score=baseline_score)

                # (5) C <- top K weighted classifiers in C U { C' }
                if len(self.models) < self.K:
                    self.models.add(value=clf_new)
                else:
                    if clf_new.weight > 0 and clf_new.weight > self.models[0].weight:
                        self.models.pop(0)
                        self.models.add(value=clf_new)

                # real shit happens only in CostSensitiveEnsemble
                self.do_instance_pruning()

        return self

    def do_instance_pruning(self):
        # do nothing here
        pass

    def predict(self, X):
        """
        Predicts the labels of X in a multiClass classification setting
        The prediction is done via weighted voting (choosing the maximum)

        :return: a list containing predictions
        """

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
                if i == len(list_label_instance): # maintain the dictionary
                    list_label_instance.append({label: weight / sum_weights})
                else:
                    try:
                        list_label_instance[i][label] += weight / sum_weights
                    except:
                        list_label_instance[i][label] = weight / sum_weights

        predict_weighted_voting = []
        for dic in list_label_instance:
            max_value = max(dic.items(), key=operator.itemgetter(1))[0] # return the key of max value in a dict
            predict_weighted_voting.append(max_value)

        return predict_weighted_voting

    def compute_MSE(self, X, y, model):
        """
        Compute the mean square error of a classifier, via the predicted probabilities.

        This code needs to take into account the fact that a classifier C trained previously
        on a chunk of data does not yield the probabilities that correspond to another chunk of data

        :param y: the true labels
        :param probabs: the predicted probability
        :param labels: the unique labels associated to the classifier that gives this predicted probability
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

        return (sum_error / N)

    def compute_weight(self, model, random_score):
        """
        Compute the weight of a classifier given the random score (calculated on a random learner).
        The weight relies on either (1) MSE if it is a normal classifier,
        or (2) benefit if it is a cost-sensitive one

        :param X: the training data
        :param y: the training labels
        :param clf: a classifier of skmultiflow (not WeightedClassifier)
        :param random_score: the baseline calculated on a random learner
        :return: the weight of clf
        """

        MSE_i = self.compute_MSE(X=self.X_chunk, y=self.y_chunk, model=model)
        return random_score - MSE_i

    def compute_random_baseline(self, classes):
        """
        This method computes the score produced by a random classifier,
        served as a baseline. The random score is MSE_r in case of a normal classifier,
        but it changes to b_r in case of a cost-sensitive one
        :return:
        """

        # assume uniform distribution
        # L = len(np.unique(classes))
        # MSE_r = L * (1 / L) * (1 - 1 / L) ** 2

        # based on the class distribution of the data
        _, class_count = np.unique(classes, return_counts=True)
        class_dist = [class_count[i] / self.S for i, c in enumerate(classes)]
        MSE_r = np.sum([class_dist[i] * ((1 - class_dist[i]) ** 2) for i, c in enumerate(classes)])
        return MSE_r
