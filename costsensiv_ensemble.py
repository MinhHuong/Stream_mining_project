import numpy as np
import heapq as hq
from skmultiflow.trees import HoeffdingTree
from skmultiflow.ensemble import WeightedEnsembleClassifier


class CostSensitiveWeightedEnsembleClassifier:
    """
    An ensemble that focuses rather on cost-sensitive ensemble classifier
    """

    class WeightedClassifier:
        """
        An inner class that eases the control of weights and crazy shit
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
            Compares an object of this class to the other, by means of its weight.
            This method helps the heap to sort the classifier correctly
            :param other: the other object of this class
            :return: true i.e. this object is smaller than the other object by means of their weight
            """
            return self.weight < other.weight

    def __init__(self, K=10, epsilon=3, cost=90):
        """
        Create a new ensemble
        :param K: the maximum number of models allowed in this ensemble
        :param epsilon: the numbers of bins
        :param cost: the real cost of a transaction
        """

        # top K classifiers
        self.K = K

        # the classifier with the minimum weight
        self.bottom_clf = None

        # a heap of weighted classifier s.t. the 1st element is the classifier with the smallest weight
        self.models = []

        # init statistics of bins (i,k) each epsilon bin has an array k
        self.bins = epsilon * [K * [{'mean': 0, 'var': 0, 'num': 0}]]

        self.cost = cost

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """
                Fit the ensemble to a data chunk
                use the basic algo 1 as described in the paper

                :param X: the training data (a data chunk S)
                :param y: the training labels
                :param classes: array-like, contains all possible labels,
                                if not provided, it will be derived from y
                :param weight:  array-like, instance weight
                                if not provided, uniform weights are assumed

                :return: self
                """

        """
        we'lll try to compute the confident prediction 
        implement algorithm 2

        we can represent fraud by 1
        and not fraud by 0
        we need a variable cost predefined

        """

        ##
        # Algo 1
        ##

        # if the classes are not provided, we derive it from y
        if classes is None:
            classes = np.unique(y)
        L = len(classes)

        # (1) train classifier C' from X
        # TODO allows a wider variety of classifiers (maybe chosen by the users?)
        C_new = HoeffdingTree()  # for now let's suppose we only train Hoeffding Tree
        C_new.partial_fit(X, y)

        # (2) compute error rate/benefit of C_new via cross-validation on S

        # MSE_r: compute the baseline error rate given by a random classifier
        # we assume that the class distribution is uniform i.e. p(c) = 1/|classes|
        # TODO later we may derive the class distribution from the labels
        p_c = 1 / L
        MSE_r = L * (p_c * (1 - p_c) ** 2)

        # MSE_i: compute the error rate of C_new via cross-validation on X
        # f_ic = the probability given by C_new that x is an instance of class c
        MSE_i = self.compute_MSE(y, C_new.predict_proba(X), classes)

        # (3) derive weight w_new for C_new using (8) or (9)
        w_new = MSE_r - MSE_i

        # create a new classifier with its associated weight,
        # the unique labels of the data chunk it is trained on
        clf_new = self.WeightedClassifier(clf=C_new, weight=w_new, chunk_labels=classes)

        # update the bottom classifier
        # if the bottom classifier has not been set, set it to the newly trained classifier
        # then push it to the heap
        if self.bottom_clf is None:
            self.bottom_clf = clf_new
            self.models.append(clf_new)

        # (4) update the weights of each classifier in the ensemble

        for i, clf in enumerate(self.models):
            MSE_i = self.compute_MSE(y, clf.clf.predict_proba(X), clf.chunk_labels)  # (1) apply Ci on S to derive MSE_i
            clf.weights = MSE_r - MSE_i  # (2) update wi based on (8) or (9)

        # (5) C <- top K weighted classifiers in C U { C' }
        # selecting top K models by dropping the worst model i.e.
        # the classifier with the smallest weight in the ensemble
        if len(self.models) < self.K:
            # just push the new model in if there is still slots
            self.models.append(clf_new)
        else:
            # if the new model has a weight > that of the bottom classifier (worst one)
            # do nothing if the new model has a weight even lower than that of the worst classifier
            if clf_new.weight > self.bottom_clf.weight:
                # we remmove the bottom one (the last one) and push the new one
                self.models.pop(len(self.models) - 1)
                self.models.append(clf_new)
                # hq.heappushpop(self.models, clf_new) # push the new classifier and remove the bottom one

        ## sort the models decrementally by their weights before outputing (the first one has the highest weight the last one is the bottom)
        self.models.sort(key=lambda clf: clf.weight, reverse=True)
        self.bottom_clf = self.models[len(self.models) - 1]  # update the bottom classifier

        # TODO Algorithm 2: do instance-based pruning here
        #####
        # here implement algorithm 2
        #####

        '''

             (1) compute Fk at each stage 
             (2) assign it to a bin 
             (3) compute the statistics : mean and variance of the error for each bin (i,k)

             compute the sum of weight of each classifier for each example seen 
        '''

        # (1) compute Fk for each example seen in X

        predict_proba_fraud = len(self.models) * [None]  # retreive the probability of predicting fraud for each model (K models)

        for i, instance in enumerate(y):

            sum_weight = 0
            current_F = 0
            F_vect = []  # Fk at each stage
            err_x = []  # error

            for k, clf in enumerate(self.models):
                sum_weight += clf.weight
                # compute the curent probability

                # if the probability is not initialized we call the predict proba method
                try:

                    if predict_proba_fraud[k] == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)

                except:
                    # it's a bit tricky to test boolean values on a  numpy array that is not empty so we use .any() method instead
                    if predict_proba_fraud[k].any() == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)

                # if we don't have the probability of predicting fraud it will be  0 so we don't do anything
                # fraud is equal to 1
                if len(predict_proba_fraud[k][i]) == 2:
                    current_F += clf.weight * predict_proba_fraud[k][i][1]

                # (2) compute the Fk for each example seen at each stage

                F_k = current_F / sum_weight
                F_vect.append(F_k)

            # (3) compute the error
            err_x = F_vect - F_vect[len(F_vect) - 1]

            # (4) update the mean and the variance of the error of these training examples for each bin (i,k)
            # we look at the error at each step for the given example
            for k, err in enumerate(err_x):

                # 1 --> we assign Fk to the corresponding bin (i,k) or (j,k)here because we used i index before
                eps = len(self.bins)
                for j in range(0, eps):

                    if F_vect[k] >= (j / eps) and F_vect[k] < ((j + 1) / eps):
                        self.bins[j][k]['num'] += 1

                        # 2-->  we compute the mean error in this bin
                        self.bins[j][k]['mean'] += err

                        # 2-->  we compute the variance of the error in this bin (basically we will just comute the squered error and do the division later)
                        self.bins[j][k]['var'] += err ** 2

                        # if we assign go to the next stage
                        break

        # after computing evry thing we do the division by the total number assigned to a bin
        for i in range(0, len(self.bins)):
            # it's a bit tricky because sometimes we can have bin's that doesn't have any input example
            # what should we do ?? --> it remains at 0
            for k in range(0, len(self.models)):
                if self.bins[i][k]['num'] > 0:
                    # divide the sum of error by the number of examples in the bin
                    self.bins[i][k]['mean'] = self.bins[i][k]['mean'] / self.bins[i][k]['num']

                    # compute the variance
                    self.bins[i][k]['var'] = (self.bins[i][k]['var'] / self.bins[i][k]['num']) - (
                    self.bins[i][k]['mean']) ** 2

        return self


    def predict(self, X, t = 3):
        """
        Algorithm 3
        :param X: the given test examples
        :param t the confidence level t
        :return: the predition
        """
        #TODO Implement Algorithm 3 here classification with instance based pruning

        # init prediction array
        prediction = []

        predict_proba_fraud = len(self.models) * [
            None]  # retreive the probability of predicting fraud for each model (K models)

        #  we do the computation for all input test examples
        for i, instance in enumerate(X):

            sum_weight = 0
            F_k = 0
            # for k in= {1,2.....K} do
            for k, clf in enumerate(self.models):

                ###
                #   (1) compute the corresponding Fk(x)
                ###

                sum_weight += clf.weight  # sum of weights

                F_k = (F_k * sum_weight) / sum_weight  # compute one part of Fk(y) with the weights

                # to compute the other part with the proba we consult th eproba first in the array index

                # if the probability is not initialized we call the predict proba method
                try:

                    if predict_proba_fraud[k] == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)

                except:
                    # it's a bit tricky to test boolean values on a  numpy array that is not empty so we use .any() method instead
                    if predict_proba_fraud[k].any() == None:
                        predict_proba_fraud[k] = clf.clf.predict_proba(X)

                # if we don't have the probability of predicting fraud it will be  0 so we don't do anything
                # fraud is equal to 1
                if len(predict_proba_fraud[k][i]) == 2:
                    F_k += (clf.weight * predict_proba_fraud[k][i][1]) / sum_weight

                # we take the amount of the transaction which we will find it in the last column feature. to apply rules (10)
                t_y = instance[len(instance) - 1]

                ###
                #   (2) we assign Fk value to a bin i or j in our case because we used i index before (the k is fixed )
                ###
                boolean = False  # i use the boolean values to remember if i find a value because using 2 times break doen't work here
                j = 0
                eps = len(self.bins)
                while j < eps and boolean == False:

                    stat = self.bins[j][k]
                    if F_k >= (j / eps) and F_k < ((j + 1) / eps):

                        # (3) apply rule (10) for this bin

                        rule_10 = F_k - stat['mean'] - t * stat['var']

                        if rule_10 > (self.cost / t_y):  # predict fraud : 1
                            boolean = True  # true we found a value
                            prediction.append(1)



                        else:
                            rule_10 = F_k + stat['mean'] + t * stat['var']

                            if rule_10 <= (self.cost / t_y):  # predict non-fraud : 0
                                boolean = True  # true we found a value
                                prediction.append(0)

                    j = j + 1

                if boolean == True:  # if  we found a value we go to the next example
                    break
            ###
            # (4) if no classifier left
            ##

            if boolean == False:
                if F_k > self.cost / t_y:
                    prediction.append(1)
                else:
                    prediction.append(0)

        return prediction

    def compute_MSE(self, y, probabs, labels):

        """
        Compute the mean square error of a classifier, via the predicted probabilities.

        It is a bit tricky here:
        - Suppose that we have a dataset D with 7 labels D_L = [1 2 3 4 5 6 7]; |D_L| = 7

        - We have a classifier C trained on a chunk of data S where |S| << |D|
        and in S we only see 4 labels S_L = [1 3 4 6] appear; |S_L| = 4

        - After being trained, C is able to predict the probability that an example
        has a label that appears in S_L i.e. the result of C.predict_proba(S)
        is an array of shape (|S|, 4) instead of (|S|, 7)

        - Now we want to use C to predict the label probability of a new chunk of data S',
        and S' may have only 2 unique labels appear in it (for example S'_L = [1 2]), so
        for an example in S', if we want to ask for the probability of the label 2, C cannot
        give it to us because it has not seen this label 2 when it is trained on the chunk S.

        If such case appears we simply set the probability of the missing label to 0

        This code needs to take into account the fact that a classifier C trained previously
        on a chunk of data does not yield the probabilities that correspond to another chunk of data

        :param y: the true labels
        :param probabs: the predicted probability
        :param labels: the unique labels associated to the classifier that gives this predicted probability
        :return: the mean square error MSE_i
        """
        N = len(y)
        sum_error = 0
        for i, c in enumerate(y):
            if c not in labels:  # if it is a label unseen when training, set the probability to 0
                probab_ic = 0
            else:  # else, find the corresponding label index
                index_label_c = np.where(labels == c)[0][
                    0]  # find the index of this label c in probabs[i] (multilabel, multiclass)
                probab_ic = probabs[i][index_label_c]
            sum_error += (1 - probab_ic) ** 2
        return (sum_error / N)