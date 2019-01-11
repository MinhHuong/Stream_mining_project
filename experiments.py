from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from ensemble import WeightedEnsembleClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

seed = 420

hyper_gen = HyperplaneGenerator(random_state=seed,
                                n_features=10,          # number of features to generate
                                n_drift_features=2,     # number of features involved in concept drift (k)
                                mag_change=0.0,         # magnitude of change (t)
                                noise_percentage=0.05,  # noise percentage (p)
                                sigma_percentage=0.1)   # probab that the direction of change is reversed (s_i)
hyper_gen.prepare_for_use()

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=20000, show_plot=True,
                                metrics=['mean_square_error'], output_file='result.csv',
                                batch_size=250)

clf = WeightedEnsembleClassifier(K=100, learner="tree")

# 4. Run
evaluator.evaluate(stream=hyper_gen, model=clf)
