from ensemble import WeightedEnsembleClassifier
from costsensiv_ensemble import CostSensitiveWeightedEnsembleClassifier
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from skmultiflow.data import HyperplaneGenerator, RandomTreeGenerator

# prepare the stream
# reuse the electricity stream for test
path_data = 'data/'

file = 'creditcard'
stream = FileStream(path_data + file +'.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

# instantiate a classifier
clf = CostSensitiveWeightedEnsembleClassifier(base_learner=HoeffdingTree(), S=500, K=10, cv=5)

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=100000, show_plot=False,
                                metrics=['accuracy', 'kappa'], output_file='result.csv',
                                batch_size=1)

# 4. Run
evaluator.evaluate(stream=stream, model=clf)
