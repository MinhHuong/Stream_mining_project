from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from ensemble import WeightedEnsembleClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees import HoeffdingTree
from skmultiflow.utils import constants
import pandas as pd
import matplotlib.pyplot as plt

seed = 420

hyper_gen = HyperplaneGenerator(random_state=seed,
                                n_features=10,          # number of features to generate
                                n_drift_features=2,     # number of features involved in concept drift (k)
                                mag_change=0.1,         # magnitude of change (t) #TODO check the t param based on the paper
                                noise_percentage=0.05,  # noise percentage (p)
                                sigma_percentage=0.1)   # probab that the direction of change is reversed (s_i)
hyper_gen.prepare_for_use()

def data_generator_different_chunk(chunk_sizes,pretrain_size=1000,max_samples=20000,K=8,learner="tree",WeightedEnsemble = True,show_plot=True):
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
		error.append(MSE)

	df['time'] = time
	df['MSE'] = error
	df['chunk'] = chunk_sizes

	print(df)
	df.to_csv(output_summary)
	return df


def data_generator_different_K(Ks,chunk_size=500,pretrain_size=1000,max_samples=20000,learner="tree",WeightedEnsemble = True,show_plot=True):
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
			clf = WeightedEnsembleClassifier(K=K, learner=learner)
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

	print(df)
	df.to_csv(output_summary)
	return df




if __name__ == '__main__':
	chunks = [200,400,600,800,1000,1200]
	Ks = [2,4,6,8,10,12]
	data_generator_different_chunk(chunk_sizes=chunks,show_plot=False,WeightedEnsemble=True)
	data_generator_different_chunk(chunk_sizes=chunks,show_plot=False,WeightedEnsemble=False)
	data_generator_different_K(Ks=Ks,show_plot=False,WeightedEnsemble=True)
	data_generator_different_K(Ks=Ks,show_plot=False,WeightedEnsemble=False)




