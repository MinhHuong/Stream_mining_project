from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from ensemble import WeightedEnsembleClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

from skmultiflow.utils import constants
import pandas as pd
import matplotlib.pyplot as plt

seed = 420

hyper_gen = HyperplaneGenerator(random_state=seed,
                                n_features=10,          # number of features to generate
                                n_drift_features=2,     # number of features involved in concept drift (k)
                                mag_change=0.1,         # magnitude of change (t)
                                noise_percentage=0.05,  # noise percentage (p)
                                sigma_percentage=0.1)   # probab that the direction of change is reversed (s_i)
hyper_gen.prepare_for_use()

# evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=50000, show_plot=True,
#                                 metrics=['mean_square_error'], output_file='result.csv',
#                                 batch_size=500)
#
# clf = WeightedEnsembleClassifier(K=100, learner="tree")
# clf_hoeff = HoeffdingTree()
#
# # 4. Run
# evaluator.evaluate(stream=hyper_gen, model=[clf, clf_hoeff],
#                    model_names=["Ensemble", "Hoeffding"])
#
# ### try decision tree ###
# clf_dt = DecisionTreeClassifier()
# hyper_gen.restart()
# num_data = 0
# sum_mse = 0
#
# # fit on some samples first
# X, y = hyper_gen.next_sample(500)
# clf_dt.fit(X, y)
# num_iter = 0
#
# while hyper_gen.has_more_samples() and num_data < 50000:
#     print("Iteration: %d, processed: %d samples" % (num_iter, num_data))
#     X, y = hyper_gen.next_sample(500)
#     sum_mse += mean_squared_error(y, clf_dt.predict(X))
#     clf_dt.fit(X, y)
#     num_data += 500
#     num_iter += 1
#
# print("Score of Decision Tree:", sum_mse / num_iter)

def data_generator_different_chunk(chunk_sizes, pretrain_size=1000, max_samples=20000,
                                   K=8, learner="tree", WeightedEnsemble = True, show_plot=True):
    df = pd.DataFrame()
    time = []
    error = []

    for chunk in chunk_sizes:
        output_file = 'experiments/'
        output_summary = 'experiments/'

        if WeightedEnsemble:
            output_file += 'ensemble/'
            output_summary += 'ensemble/'
        else:
            output_file += 'single/'
            output_summary += 'single/'

        output_file += 'diff_chunk/'
        output_summary += 'diff_chunk/summary/'

        output_file+= 'chunk'+str(chunk)

        output_file+= '_K'+str(K)
        output_summary += 'K'+str(K)

        output_file+= '_MAXSample'+str(max_samples)
        output_summary+= '_MAXSample'+str(max_samples)
        output_summary += '_summary.csv'

        evaluator = EvaluatePrequential(pretrain_size=pretrain_size, max_samples=max_samples, show_plot=show_plot,
		                                metrics=['mean_square_error'], output_file=output_file+'.csv',
		                                batch_size=chunk)

        if WeightedEnsemble:
            clf = WeightedEnsembleClassifier(K=K, learner=learner)
        else:
            clf = HoeffdingTree() #TODO

        # 4. Run
        evaluator.evaluate(stream=hyper_gen, model=clf)
        time.append(evaluator._end_time-evaluator._start_time)
        MSE = evaluator._data_buffer.get_data(metric_id=constants.MSE, data_id=constants.MEAN)[-1]
        MSE *= 100 #%
        # error.append(MSE)

    df['time'] = time
    df['MSE'] = error
    df['chunk'] = chunk_sizes

    print(df)
    df.to_csv(output_summary)
    return df

def data_generator_different_K(Ks, chunk_size=500, pretrain_size=1000,
                               max_samples=20000, learner=DecisionTreeClassifier(),
                               WeightedEnsemble=True, show_plot=True):
    # single Gk is wrong right now, there is no K for single Gk
    df = pd.DataFrame()
    time = []
    error = []
    for K in Ks:
        output_file = 'experiments/'
        output_summary = 'experiments/'

        if WeightedEnsemble:
            output_file += 'ensemble/'
            output_summary += 'ensemble/'
        else:
            output_file += 'single/'
            output_summary += 'single/'

        output_file += 'diff_k/'
        output_summary += 'diff_k/summary/'

        output_file+= 'K'+str(K)

        output_file+= '_chunk'+str(chunk_size)
        output_summary += 'chunk'+str(chunk_size)

        output_file+= '_MAXSample'+str(max_samples)
        output_summary+= '_MAXSample'+str(max_samples)
        output_summary += '_summary.csv'

        evaluator = EvaluatePrequential(pretrain_size=pretrain_size, max_samples=max_samples, show_plot=show_plot,
		                                metrics=['mean_square_error'], output_file=output_file+'.csv',
		                                batch_size=chunk_size)

        if WeightedEnsemble:
            clf = WeightedEnsembleClassifier(K=K, base_learner=learner)
        else:
            clf = HoeffdingTree() #TODO --> like 6.1 paper

        # 4. Run
        evaluator.evaluate(stream=hyper_gen, model=clf)
        time.append(evaluator._end_time-evaluator._start_time)
        MSE = evaluator._data_buffer.get_data(metric_id=constants.MSE, data_id=constants.MEAN)[-1]
        MSE *= 100 #%
        error.append(MSE)

    df['time'] = time
    df['MSE'] = error
    df['K'] = Ks

    # print(df)
    df.to_csv(output_summary)

    return df


if __name__ == '__main__':
    chunks = range(300, 1100, 100)
    Ks = [1] + list(range(2, 51, 1))
    # data_generator_different_chunk(chunk_sizes=chunks,show_plot=False,WeightedEnsemble=True)
    # data_generator_different_chunk(chunk_sizes=chunks,show_plot=False,WeightedEnsemble=False)

    for chunk in chunks:
        data_generator_different_K(Ks=Ks, chunk_size=chunk, show_plot=False, WeightedEnsemble=True)

    # data_generator_different_K(Ks=Ks,show_plot=False,WeightedEnsemble=False)
