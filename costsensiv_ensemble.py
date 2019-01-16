import numpy as np
from skmultiflow.trees import HoeffdingTree
from ensemble import WeightedEnsembleClassifier


class CostSensitiveWeightedEnsembleClassifier(WeightedEnsembleClassifier):
    """
    An ensemble that focuses rather on cost-sensitive ensemble classifier
    """

    def __init__(self, K=10, learner="tree", epsilon=3, cost=90):
        """
        Create a new ensemble
        :param K: the maximum number of models allowed in this ensemble
        :param epsilon: the numbers of bins
        :param cost: the real cost of a transaction
        """

        # let the parent do the stuff
        super().__init__(K=K, learner=learner)

        # init statistics of bins (i,k) each epsilon bin has an array k
        self.bins = epsilon * [K * [{'mean': 0, 'var': 0, 'num': 0}]]

        # a cost (because it is a cost sensitive classifier)
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

         # call Algorithm (1) as the base
        super().partial_fit(X=X, y=y, classes=classes, weight=weight)

        """
        We'll try to compute the confident prediction 
        implement algorithm 2

        we can represent fraud by 1
        and not fraud by 0
        we need a variable cost predefined
        """

        '''
        (1) compute Fk at each stage 
        (2) assign it to a bin 
        (3) compute the statistics : mean and variance of the error for each bin (i,k)

        compute the sum of weight of each classifier for each example seen 
        '''

        # (1) compute Fk for each example seen in X

        # retrieve the probability of predicting fraud for each model (K models)
        predict_proba_fraud = len(self.models) * [None]

        for i, instance in enumerate(y):  # for each y in S do
            # here compute the benifit call befinit call

            sum_weight = 0

            current_F = 0
            F_vect = np.zeros(self.K) # Fk at each stage
            err_x = np.zeros(self.K)  # error at each stage k in K

            # compute F_k(y) for k = 1,...,K
            k = -1
            for model in self.models.islice(start=0, stop=self.K, reverse=True):
                k += 1

                clf = model.clf
                sum_weight += model.weight
                # print("Weight:", model.weight)

                # compute the curent probability
                # if the probability is not initialized we call the predict proba method
                try:
                    if predict_proba_fraud[k] == None:
                        predict_proba_fraud[k] = clf.predict_proba(X)
                except:
                    # it's a bit tricky to test boolean values on a  numpy array
                    # that is not empty so we use .any() method instead
                    if predict_proba_fraud[k].any() == None:
                        predict_proba_fraud[k] = clf.predict_proba(X)

                # if we don't have the probability of predicting fraud it will be  0 so we don't do anything
                # fraud is equal to 1
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

                    if F_vect[k] >= (j / eps) and F_vect[k] < ((j + 1) / eps):
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
            # it's a bit tricky because sometimes we can have bin's that doesn't have any input example
            # what should we do ?? --> it remains at 0
            for k in range(self.K):
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
                # print(k, clf.weight)

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
                    if F_k >= (j / eps) and F_k < ((j + 1) / eps):
                        # (3) apply rule (10) for this bin

                        # Huong: I tidied the code a bit here
                        if F_k - stat['mean'] - t * stat['var'] > (self.cost / t_y): # FRAUD
                            boolean = True
                            prediction[i] = 1
                        elif F_k + stat['mean'] + t * stat['var'] <= (self.cost / t_y): # NON-FRAUD
                            boolean = True
                            prediction[i] = 0

                        # rule_10 = F_k - stat['mean'] - t * stat['var']
                        #
                        # if rule_10 > (self.cost / t_y):  # predict fraud : 1
                        #     boolean = True  # true we found a value
                        #     prediction.append(1)
                        # else:
                        #     rule_10 = F_k + stat['mean'] + t * stat['var']
                        #
                        #     if rule_10 <= (self.cost / t_y):  # predict non-fraud : 0
                        #         boolean = True  # true we found a value
                        #         prediction.append(0)

                    j = j + 1

                if boolean:  # if we found a value we go to the next example
                    break

            # (4) if no classifier left

            # Huong: I change the condition here a bit:
            # if no classifier left i.e. we have consulted every clf in the ensemble
            # without obtaining a certain answer --> prediction[i] is not yet given
            # so we just need to check whether prediction[i] is -1
            if prediction[i] == -1:
                if F_k > self.cost / instance[-1]:  # instance[-1] is just t(y)
                    prediction[i] = 1
                else:
                    prediction[i] = 0

        return prediction


    def compute_benefit(self, X, y, probabs, labels):
        """

        Computes the benefit
        :param y is the real class
        :return:
        """


        sum_benefit = 0

        for i, c in enumerate(y):
            # c is the real label
            # if the label in y is unseen when training, skip it, don't include it in the error
            # 0 --> not fraud
            # 1 --> fraud


            for j, cprime in enumerate(labels) :

                # (1) compute the benefit matrix
                benefit_c_cprime = 0
                if c == 1 : # if it's the actual fraud == 1
                    if cprime == 1 :
                        # the amount - the
                        benefit_c_cprime = X[i][-1] - self.cost
                else : # if it's the actual not fraud == 0
                    if cprime == 1 :
                        benefit_c_cprime= - self.cost


                # (2) get the probability
                index_label_cprime = np.where(labels == cprime)[0][0]  # find the index of this label c in probabs[i]
                probab_ic = probabs[i][index_label_cprime]

                sum_benefit += probab_ic * benefit_c_cprime




        return sum_benefit

    
    def compute_weight(self, X, y, clf, random_score):
        """
         Overrides the function defined in the parent class WeightedEnsembleClassifier,
        such that the weight is now computed by the benefits and not MSE

        :param X:
        :param y:
        :param clf:
        :param random_score:
        :return:
        """


        b_i = self.compute_benefit( X, y, probabs=clf.clf.predict_proba(X), labels=clf.chunk_labels)
        return b_i - random_score


    def compute_random_baseline(self, classes, class_count, size):
        """
        Overrides the function defined in the parent class WeightedEnsembleClassifier,
        such that the baseline is now computed by the benefit of a random classifier

        :param classes:
        :param class_count:
        :param size:
        :return:
        """

        # based on the class distribution of the data
        if class_count is None:
            _, class_count = np.unique(classes, return_counts=True)
        class_dist = [class_count[i] / size for i, c in enumerate(classes)]
        MSE_r = np.sum([class_dist[i] * ((1 - class_dist[i]) ** 2) for i, c in enumerate(classes)])
        # return MSE_r

        return 0