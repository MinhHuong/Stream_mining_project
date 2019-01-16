from ensemble import WeightedEnsembleClassifier
from costsensiv_ensemble import CostSensitiveWeightedEnsembleClassifier
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# prepare the stream
# reuse the electricity stream for test
path_data = 'data/'

File = 'elec'
# File = 'creditcard'
stream = FileStream(path_data + File +'.csv', n_targets=1, target_idx=-1)
stream.prepare_for_use()

# instantiate a classifier
# clf = CostSensitiveWeightedEnsembleClassifier()
clf = WeightedEnsembleClassifier(K=100, base_learner=DecisionTreeClassifier())
# h = [clf, HoeffdingTree()]


evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=50000, show_plot=True,
                                metrics=['mean_square_error'], output_file='result.csv',
                                batch_size=250)

# 4. Run
evaluator.evaluate(stream=stream, model=clf)