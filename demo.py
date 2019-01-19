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

File = 'elec'
# File = 'creditcard'
# stream = FileStream(path_data + File +'.csv', n_targets=1, target_idx=-1)
# stream.prepare_for_use()

hyper_gen = HyperplaneGenerator(random_state=420,
                                n_features=10,          # number of features to generate
                                n_drift_features=2,     # number of features involved in concept drift (k)
                                mag_change=0.1,         # magnitude of change (t)
                                noise_percentage=0.05,  # noise percentage (p)
                                sigma_percentage=0.1)   # probab that the direction of change is reversed (s_i)
hyper_gen.prepare_for_use()

stream = RandomTreeGenerator()
stream.prepare_for_use()

# instantiate a classifier
# clf = CostSensitiveWeightedEnsembleClassifier()
clf = WeightedEnsembleClassifier(K=10, base_learner=HoeffdingTree())
# h = [clf, HoeffdingTree()]


evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=100000, show_plot=True,
                                metrics=['accuracy', 'kappa'], output_file='result.csv',
                                batch_size=500)

# File = 'elec'
File = 'creditcard'
stream = FileStream(path_data + File +'.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

# instantiate a classifier
clf = CostSensitiveWeightedEnsembleClassifier()
#clf = WeightedEnsembleClassifier(K=10, learner="tree")

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=50000, show_plot=True,
                                metrics=['accuracy', 'kappa'], output_file='result.csv',
                                batch_size=100)

# 4. Run
evaluator.evaluate(stream=stream, model=clf)