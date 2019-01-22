#!/usr/bin/env python

"""costsensiv_ensemble.py: An extension of Weighted Ensemble Classifier for cost-sensitive applications"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ensemble import WeightedEnsembleClassifier
from sklearn.model_selection import StratifiedKFold
import copy as cp


__author__ = "Armita KHAJEHNASSIRI, Soumeya KAADA, Minh-Huong LE-NGUYEN"


class CostSensitiveWeightedEnsembleClassifier(WeightedEnsembleClassifier):
    """An extension of weighted ensemble classifier
    that focuses rather on cost-sensitive applications.
    This ensemble classifier only works with BINARY classification (for now).

    The implementation follows the techniques described in
    "Mining Concept-Drifting Data Streams using Ensemble Classifiers",
    by Haixun Wang, Wei Fan, Philip S. Yu, Jiawei Han.
    """

    def __init__(self, K=10, base_learner=DecisionTreeClassifier(), S=200, cv=5,
                 epsilon=3, cost=90, fraud_label=1, t=3):
        """
        Initializes a cost-sensitive weighted ensemble.

        :param K: the number of classifiers in the ensemble
        :param base_learner: the base learner
        :param S: chunk size
        :param epsilon: number of bins to do instance-based pruning
        :param cost: the cost threshold to determine whether to launch an investigation
        :param fraud_label: the label indicating the fraud
        :param cv: number of cross-validation folds to compute the score of a new classifier
        :param t: confidence level
        """

        # let the parent do the stuffs
        super().__init__(K=K, base_learner=base_learner, S=S, cv=cv)

        # init statistics of bins (i,k) each epsilon bin has an array k
        self.bins = epsilon * [K * [{'mean': 0, 'var': 0, 'num': 0}]]

        # a cost (because it is a cost sensitive classifier)
        self.cost = cost

        # the fraud label as indicated by the users
        self.fraud_label = fraud_label

        # confidence level
        self.t = t

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """
        Fits the ensemble to a data chunk. It inherits the Algorithm 1 as described in the paper
        to update the ensemble, then launches an instance-based pruning process at the end.

        :param X: the training data
        :param y: the training labels
        :param classes: array-like, contains all possible labels, derived from y if not provided
        :param weight:  array-like, instance weight, uniform weights are assumed if not provided
        :return: self
        """

        # call Algorithm (1) as the base
        # the function `do_instance_pruning` will be launched afterwards
        return super().partial_fit(X=X, y=y, classes=classes, weight=weight)

    def do_instance_pruning(self):
        """
        Does instance-based pruning as described in Algorithm  in the paper

        (1) Compute Fk at each stage
        (2) Assign it to a bin
        (3) Compute the statistics: mean and variance of the error for each bin (i, k)

        Compute the sum of weight of each classifier for each example seen
        """

        # retrieve the probability of predicting fraud for each model (K models)
        predict_proba_fraud = len(self.models) * [None]

        # for each instance in the data chunk
        for i, instance in enumerate(self.y_chunk):
            sum_weight = 0
            current_F = 0
            F_vect = np.zeros(self.K)  # Fk at each stage

            # compute F_k(y) for k = 1...K - the classifiers are sorted in DESCENDING order of weights
            k = -1
            for model in self.models.islice(start=0, stop=self.K, reverse=True):
                k += 1
                clf = model.clf
                sum_weight += model.weight

                # compute the current probability
                # if the probability is not initialized we call the `predict_proba` method
                try:
                    if predict_proba_fraud[k] == None:
                        predict_proba_fraud[k] = clf.predict_proba(self.X_chunk)
                except:
                    # it's a bit tricky to test boolean values on a  numpy array
                    # that is not empty so we use .any() method instead
                    if predict_proba_fraud[k].any() == None:
                        predict_proba_fraud[k] = clf.predict_proba(self.X_chunk)

                # if we don't have the probability of predicting fraud it will be 0 so we don't do anything
                if len(predict_proba_fraud[k][i]) == 2:
                    current_F += model.weight * predict_proba_fraud[k][i][1]

                # (2) compute the Fk for each example seen at each stage
                F_k = current_F / sum_weight
                F_vect[k] = F_k

            # (3) compute the error
            err_x = F_vect - F_vect[-1]

            # (4) update the mean and the variance of the error of these training examples for each bin (i,k)
            # we look at the error at each step for the given example
            for k, err in enumerate(err_x):

                # 1 --> we assign Fk to the corresponding bin (i,k) or (j,k)here because we used i index before
                eps = len(self.bins)
                for j in range(0, eps):

                    if (j / eps) <= F_vect[k] < ((j + 1) / eps):
                        self.bins[j][k]['num'] += 1

                        # 2-->  we compute the mean error in this bin
                        self.bins[j][k]['mean'] += err

                        # 2-->  we compute the variance of the error in this bin
                        # (basically we will just compute the squared error and do the division later)
                        self.bins[j][k]['var'] += err ** 2

                        # if we assign go to the next stage
                        break

        # after computing everything we do the division by the total number assigned to a bin
        for i in range(0, len(self.bins)):
            # it's a bit tricky because sometimes we can have bins that don't have any input example
            # --> it remains at 0
            for k in range(self.K):
                if self.bins[i][k]['num'] > 0:
                    # divide the sum of error by the number of examples in the bin
                    self.bins[i][k]['mean'] = self.bins[i][k]['mean'] / self.bins[i][k]['num']

                    # compute the variance
                    self.bins[i][k]['var'] = (self.bins[i][k]['var'] / self.bins[i][k]['num']) - \
                                             (self.bins[i][k]['mean']) ** 2

    def predict(self, X):
        """
        We rely on the Algorithm 3 in the paper to issue predictions.
        The idea is to ask every classifier for the predicted label and compute
        the confidence level we have every time we ask a classifier in the ensemble.
        The process stops if we are confident enough and do not need to consult more classifiers.
        But if there is no more classifier in the ensemble, we return the current prediction.
        
        :param X: the given test examples
        :return: the prediction (array of shape X.shape[1])
        """

        N, D = X.shape

        # init prediction array
        prediction = np.array([-1] * N)

        # retrieve the probability of predicting fraud for each model (K models)
        predict_proba_fraud = len(self.models) * [None]

        #  we do the computation for all input test examples
        for i, instance in enumerate(X):
            sum_weight = 0
            F_k = 0

            # for k in= {1,2.....K} do
            k = -1
            for clf in self.models.islice(start=0, stop=self.K, reverse=True):
                k += 1
                # (1) compute the corresponding Fk(x)
                sum_weight += clf.weight  # sum of weights

                # compute one part of Fk(y) with the weights (be careful: sum_weight may be 0)
                F_k = (F_k * sum_weight) / sum_weight if sum_weight != 0 else 0

                # to compute the other part with the proba we consult th eproba first in the array index
                # if the probability is not initialized we call the predict proba method
                try:
                    if predict_proba_fraud[k] == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)
                except:
                    # it's a bit tricky to test boolean values on a numpy array
                    # that is not empty so we use .any() method instead
                    if predict_proba_fraud[k].any() == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)

                # if we don't have the probability of predicting fraud it will be  0 so we don't do anything
                # fraud is equal to 1
                if len(predict_proba_fraud[k][i]) == 2:
                    F_k += (clf.weight * predict_proba_fraud[k][i][1]) / sum_weight

                # we take the amount of the transaction
                # which we will find it in the last column feature. to apply rules (10)
                t_y = instance[-1]

                # (2) we assign Fk value to a bin i (or j in our case)
                # because we used i index before (k is fixed)

                # i use the boolean values to remember
                # if i find a value because using 2 times break doesn't work here
                boolean = False
                j = 0
                eps = len(self.bins)

                # while we haven't found the bin AND no prediction has not yet been given
                while j < eps and not boolean:
                    stat = self.bins[j][k]

                    # find the bin i y belongs to
                    if (j / eps) <= F_k < ((j + 1) / eps):
                        # (3) apply rule (10) for this bin
                        # What if the amount is 0 ?
                        if t_y != 0:
                            if F_k - stat['mean'] - self.t * stat['var'] > (self.cost / t_y):  # FRAUD
                                boolean = True
                                prediction[i] = 1
                            elif F_k + stat['mean'] + self.t * stat['var'] <= (self.cost / t_y):  # NON-FRAUD
                                boolean = True
                                prediction[i] = 0
                        else:
                            boolean = True
                            prediction[i] = 0

                    j = j + 1

                if boolean:  # if we found a value we go to the next example
                    break

            # (4) if no classifier left
            # if no classifier left i.e. we have consulted every classifier in the ensemble
            # without obtaining a certain answer --> prediction[i] is not yet given
            # so we just need to check whether prediction[i] is -1
            if prediction[i] == -1:
                if instance[-1] != 0 and F_k > self.cost / instance[-1]:  # instance[-1] is just t(y)
                    prediction[i] = 1
                else:
                    prediction[i] = 0

        return prediction

    def compute_score(self, model, X, y):
        """
        Overrides the method `compute_score` of the parent.
        It computes the benefits instead of MSE

        :param model: the learner
        :param X: the data
        :param y: the label
        :return: a benefit score
        """
        sum_benefit = 0
        probabs = model.clf.predict_proba(self.X_chunk)
        chunk_classes, chunk_classes_count = np.unique(self.y_chunk, return_counts=True)

        for i, c in enumerate(self.y_chunk):
            # c is the real label
            # if the label in y is unseen when training, skip it, don't include it in the error
            # 0 --> not fraud
            # 1 --> fraud

            for j, cprime in enumerate(chunk_classes) :

                # (1) compute the benefit matrix
                benefit_c_cprime = 0
                if c == self.fraud_label :  # if it's the actual fraud == 1
                    if cprime == self.fraud_label:
                        benefit_c_cprime = self.X_chunk[i][-1] - self.cost
                else:
                    if cprime == self.fraud_label :
                        benefit_c_cprime= - self.cost

                probab_ic = 0
                if cprime in model.chunk_labels:
                    try:
                        probab_ic = probabs[i][list(model.chunk_labels).index(cprime)]
                    except IndexError:
                        probab_ic = probabs[i][np.argmax(chunk_classes_count)]

                sum_benefit += probab_ic * benefit_c_cprime

        return sum_benefit

    def compute_weight(self, model, random_score, cv=None):
        """
        Overrides the function defined in the parent class WeightedEnsembleClassifier

        :param model: the learner
        :param random_score: the baseline score
        :param cv: number of folds to do cross-validation
        :return:
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

        # w = b_i - b_r
        return score - random_score

    def compute_random_baseline(self, classes):
        """
        Overrides the function defined in the parent class WeightedEnsembleClassifier,
        such that the baseline is now computed by the benefit of a random classifier

        :param classes: number of unique class labels
        :return: the baseline score computed on a random classifier
        """

        # based on the class distribution of the data
        sum_benefit = 0

        for i, c in enumerate(self.y_chunk):
            # c is the real label
            # if the label in y is unseen when training, skip it, don't include it in the error
            # 0 --> not fraud
            # 1 --> fraud

            for j, cprime in enumerate(classes):

                # (1) compute the benefit matrix
                benefit_c_cprime = 0
                if c == 1:  # if it's the actual fraud == 1
                    if cprime == self.fraud_label:
                        # the amount - the
                        benefit_c_cprime = self.X_chunk[i][-1] - self.cost
                else:   # if it's the actual not fraud == 0
                    if cprime == self.fraud_label:
                        benefit_c_cprime = - self.cost

                # (2) get the probability
                probab_ic = 1 / len(classes)
                sum_benefit += probab_ic * benefit_c_cprime

        return sum_benefit
