"""Run different experiments"""

from ensemble import WeightedEnsembleClassifier
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
import numpy as np
import time as tm

def expe_chunks_ks_accu_kappa_time_awe(stream):
    """
    Runs the experiments that varies K, ChunkSize and measures Accuracy + Kappa + Training time.
    :return:
    """
    # chunks = range(200, 1100, 100)  # chunk size from 200 to 1000
    chunks = [250, 500, 750, 1000]

    # Ks = range(5, 21, 1)  # K from 1 to 20
    Ks = [5, 10,15]

    # max_samples = 200000
    max_samples = 100000

    # prefix = "experiments/awe/"
    prefix = "experiments/cv/"

    time_outputfile = "expe_maxsamples" + str(max_samples) + "_time_nocv.csv"
    # fw = open(prefix + time_outputfile, "w")
    # fw.write("S,k,time" + "\n")

    with open(prefix + time_outputfile, "w") as f:
        f.write("S,k,time" + "\n")
        for ik, k in enumerate(Ks):
            for ic, c in enumerate(chunks):
                print("\nK = %d and S = %d" % (k, c))
                outputfile = "k" + str(k) + "_chunk" + str(c) + "_cv0" \
                             + "_maxsamples" + str(max_samples) + "_accukappa.csv"
                evaluator = EvaluatePrequential(pretrain_size=c, max_samples=max_samples, show_plot=False,
                                                metrics=['accuracy', 'kappa'], output_file=prefix + outputfile,
                                                batch_size=1, restart_stream=True)
                clf = WeightedEnsembleClassifier(K=k, base_learner=HoeffdingTree(), S=c, cv=None)

                # run the experiments
                start = tm.time()
                evaluator.evaluate(stream=stream, model=clf)
                end = tm.time()

                elapse = end - start
                f.write(str(c) + "," + str(k) + "," + str(elapse) + "\n")
    # fw.close()

def expe_cross_validation(stream):
    """
    Runs the experiments to check the improvement with cross-validation.
    K = [5, 10, 15]
    cv = [None, 5, 10]
    chunks = [250, 500, 750, 1000]
    :param stream:
    :return:
    """
    chunks = [250, 500, 750, 1000]
    Ks = [5, 10, 15]
    CVs = [3, 5, 10]
    prefix = "experiments/cv/"
    max_samples = 100000

    time_outputfile = "experiments/cv/expe_cv_maxsamples" + str(max_samples) + "_time.csv"
    with open(time_outputfile, "w") as f:
        f.write("S,k,cv,time" + "\n")

        for s in chunks:
            for k in Ks:
                for cv in CVs:
                    print("\nK = %d, S = %d, CV = %d" % (k, s, cv))
                    outputfile = "k" + str(k) + "_chunk" + str(s) + "_cv" + str(cv) \
                                 + "_maxsamples" + str(max_samples) + "_accukappa.csv"
                    print("Experiment result in " + (prefix + outputfile))
                    evaluator = EvaluatePrequential(pretrain_size=s, max_samples=max_samples, show_plot=False,
                                                    metrics=['accuracy', 'kappa'], output_file=prefix + outputfile,
                                                    batch_size=1, restart_stream=True)
                    clf = WeightedEnsembleClassifier(K=k, base_learner=HoeffdingTree(), S=s, cv=cv)

                    # run the experiments
                    start = tm.time()
                    evaluator.evaluate(stream=stream, model=clf)
                    end = tm.time()
                    elapse = end - start

                    f.write(str(s) + "," + str(k) + "," + str(cv) + "," + str(elapse) + "\n")


if __name__ == "__main__":
    # prepare the stream
    print("\nPreparing the stream...")
    stream = HyperplaneGenerator(random_state=0,
                                 n_features=10,  # number of features to generate
                                 n_drift_features=2,  # number of features involved in concept drift (k)
                                 mag_change=0.1,  # magnitude of change (t)
                                 noise_percentage=0.05,  # noise percentage (p)
                                 sigma_percentage=0.1)  # probab that the direction of change is reversed (s_i)
    stream.prepare_for_use()

    # run some experiments here
    print("\nExperiment: Accuracy + Kappa + Training time on varying K and S, no CV...")
    expe_chunks_ks_accu_kappa_time_awe(stream)

    # print("\nExperiment: Accuracy + Kappa + Training time on varying K, S, and folds")
    # expe_cross_validation(stream)