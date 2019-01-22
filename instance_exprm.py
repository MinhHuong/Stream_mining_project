from costsensiv_ensemble import CostSensitiveWeightedEnsembleClassifier
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from skmultiflow.data import HyperplaneGenerator, RandomTreeGenerator
import numpy as np
import csv
import matplotlib.pyplot as plt

def make_filename(K, chunk, max_sample, get_time=False, ext=".csv"):
    if not get_time:
        return "K" + str(K) + "_chunk" + str(chunk) + "_MAXSample" + str(max_sample) + ext
    return "summary/chunk" + str(chunk) + "_MAXSample" + str(max_sample) + "_summary" + ext


def get_total_benefit(prefix, ks, chunks, num_chunks):
    mean_times = np.zeros(len(chunks))

    for i, chunk in enumerate(chunks):
        max_sample = chunk * num_chunks
        filename = prefix + make_filename(chunk=chunk, K=None, max_sample=max_sample, get_time=True)
        try:
            df = pd.read_csv(filename, sep=",")
            mean_times[i] = np.mean(df["time"])
        except FileNotFoundError:
            mean_times[i] = 0

    return mean_times


def pruning_method( K = 32 , step=1, chunk_size=500, pretrain_size=1000,
                               max_samples=20000, base_learner=DecisionTreeClassifier()):
    path_data = 'data/'
    file = 'creditcard'
    stream = FileStream(path_data + file + '.csv', n_targets=1, target_idx=-1)
    stream.prepare_for_use()


    # compute the benefit with instance based pruning
    total_benefit_with_IBP = []
    avg_k = []
    for k in range(20, K+1, step):
        clf_with_IBP = CostSensitiveWeightedEnsembleClassifier(K=k,base_learner=base_learner, cost = 50 )
        output_file = make_filename(k, chunk = chunk_size,  max_sample=max_samples)

        evaluator = EvaluatePrequential(pretrain_size=pretrain_size, max_samples=max_samples, show_plot=False,
                                        metrics=['accuracy', 'kappa'], output_file= output_file + '.csv',
                                        batch_size=chunk_size)
        stream.prepare_for_use()
        evaluator.evaluate(stream=stream, model=clf_with_IBP)

        # get the total benefit foreach run
        total_benefit_with_IBP.append(clf_with_IBP.total_benefit)
        avg_k.append(clf_with_IBP.avg_k_used/(max_samples-pretrain_size))

        a = np.array([clf_with_IBP.avg_k_used/(max_samples-pretrain_size)]+ [clf_with_IBP.total_benefit])
        with open('pruning_method.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(a)

    # np.savetxt("pruning_method.csv", a, delimiter=",", fmt = '%10.3f')



    return None


def No_pruning_method(K=32, step=1, chunk_size=500, pretrain_size=1000,
                   max_samples=20000, base_learner=DecisionTreeClassifier()):
    path_data = 'data/'
    file = 'creditcard'
    stream = FileStream(path_data + file + '.csv', n_targets=1, target_idx=-1)
    stream.prepare_for_use()

    # compute the benefit with instance based pruning


    for k in range(1, K + 1, step):
        clf_without_IBP = CostSensitiveWeightedEnsembleClassifier(K=k,base_learner=base_learner, do_class_with_IBP=False, cost = 50)
        output_file = make_filename(k, chunk=chunk_size, max_sample=max_samples)

        evaluator = EvaluatePrequential(pretrain_size=pretrain_size, max_samples=max_samples, show_plot=False,
                                        metrics=['accuracy', 'kappa'], output_file=output_file + '.csv',
                                        batch_size=chunk_size)
        stream.prepare_for_use()
        evaluator.evaluate(stream=stream, model=clf_without_IBP)

        # get the total benefit foreach run

        a = np.array([k] + [clf_without_IBP.total_benefit])
        with open('No_pruning_method.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(a)


    return None



def plot_IBR(name1, name2):
    with open(str(name1) + '.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')

        X_axis1 = []
        Y_axis1= []
        for row in  csv_reader:
            if row != [] :
                X_axis1.append(float("%.5f" % float(row[0])))
                Y_axis1.append(float("%.5f" % float(row[1])))
    with open(str(name2) + '.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')
        X_axis2 = []
        Y_axis2= []
        for row in  csv_reader:
            if row != [] :
                X_axis2.append(float("%.5f" % float(row[0] )))
                Y_axis2.append(float("%.0f" % float(row[1])))




    plt.figure(figsize=(7, 5))
    plt.plot(X_axis1,Y_axis1, marker='o', label="Instance-based Pruning")
    plt.plot(X_axis2,Y_axis2, marker='^', label="Classifier Ensemble Ek")

    plt.xticks(X_axis2)
    plt.xlabel("# of Classifiers in the ensembles")
    plt.ylabel("Benefits ($)")
    plt.title("reduction of ensemble size by insatnce-based pruning")
    plt.legend()
    plt.show()



def plot_IBR_BARS(name1, name2):
    with open(str(name1) + '.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')

        X_axis1 = []

        for row in  csv_reader:
            if row != [] :
                X_axis1.append(float("%.5f" % float(row[0])))

    with open(str(name2) + '.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')
        X_axis2 = []

        for row in  csv_reader:
            if row != [] :
                X_axis2.append(float("%.5f" % float(row[0] )))
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    index = np.arange(1,33,2)
    index2 = np.arange(1, 66,2)

    ax.bar(index  , np.take(X_axis1, index), width=0.9,align='edge', label="Instance-based Pruning")
    ax.bar(index ,np.take(X_axis2, index),width=-0.9, align='edge', label="Classifier Ensemble Ek")

    plt.xticks(index)
    plt.yticks(index)
    plt.xlabel("# of Classifiers in the ensembles")
    plt.ylabel("# of Classifiers actually used")
    plt.title("reduction of ensemble size by insatnce-based pruning")
    plt.legend()
    
    plt.show()








# pruning_method()
# plot_IBR('pruning_method')

#No_pruning_method()


# plot_IBR('pruning_method','No_pruning_method')

plot_IBR_BARS('pruning_method','No_pruning_method')