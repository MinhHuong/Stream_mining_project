from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from ensemble import WeightedEnsembleClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

seed = 420

hyper_gen = HyperplaneGenerator(random_state=seed,
                                n_features=10,          # number of features to generate
                                n_drift_features=2,     # number of features involved in concept drift (k)
                                mag_change=0.1,         # magnitude of change (t)
                                noise_percentage=0.05,  # noise percentage (p)
                                sigma_percentage=0.1)   # probab that the direction of change is reversed (s_i)
hyper_gen.prepare_for_use()

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=50000, show_plot=True,
                                metrics=['mean_square_error'], output_file='result.csv',
                                batch_size=500)

clf = WeightedEnsembleClassifier(K=100, learner="tree")
clf_hoeff = HoeffdingTree()

# 4. Run
evaluator.evaluate(stream=hyper_gen, model=[clf, clf_hoeff],
                   model_names=["Ensemble", "Hoeffding"])

### try decision tree ###
clf_dt = DecisionTreeClassifier()
hyper_gen.restart()
num_data = 0
sum_mse = 0

# fit on some samples first
X, y = hyper_gen.next_sample(500)
clf_dt.fit(X, y)
num_iter = 0

while hyper_gen.has_more_samples() and num_data < 50000:
    print("Iteration: %d, processed: %d samples" % (num_iter, num_data))
    X, y = hyper_gen.next_sample(500)
    sum_mse += mean_squared_error(y, clf_dt.predict(X))
    clf_dt.fit(X, y)
    num_data += 500
    num_iter += 1

print("Score of Decision Tree:", sum_mse / num_iter)