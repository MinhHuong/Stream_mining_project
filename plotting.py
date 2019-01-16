
import pandas as pd
import matplotlib.pyplot as plt
import sys

def gen_filename(value, MAXSample,folder_name, ensemble=True):
	#experiments/ensemble/chunk2000_K5_MAXSample10000_summary.csv
	file_name = 'experiments/'
	if ensemble:
		file_name += 'ensemble/'
	else:
		file_name += 'single/'

	if folder_name == 'diff_chunk':
		file_name += 'diff_chunk/summary/'
		file_name+= 'K'+str(value)+'_'

	elif folder_name == 'diff_k':
		file_name += 'diff_k/summary/'
		file_name+= 'chunk'+str(value)+'_'


	file_name+='MAXSample'+str(MAXSample)+'_'
	file_name+= 'summary.csv'

	return file_name

def draw_plot_diff_chunks_time(df_ensemble,df_single): # Figure 5: Training Time, ChunkSize, and Error Rate
	fig, ax1 = plt.subplots(figsize=(7, 6))

	ax1.plot(df_ensemble['chunk'], df_ensemble['time'], label="Training Time of E100", color="#1f77b4", marker="^")
	ax1.plot(df_single['chunk'], df_single['time'], label="Training Time of G100", linestyle="--", color="#1f77b4")
	ax1.set_xticks(df_ensemble['chunk'])
	ax1.set_xlabel('Chunk')
	ax1.set_ylabel("Trainig Time (s)")
	#ax1.set_ylim((0.0, 1.0))
	#ax1.set_yticks(np.arange(0.0, 1.0, 0.1))

	ax2 = ax1.twinx()
	ax2.plot(df_ensemble['chunk'], df_ensemble['MSE'], label="Ensemble Error Rate", linestyle=":", marker="o", color="#7f7f7f", linewidth=3.)
	ax2.set_ylabel("Error Rate (%)")

	fig.legend()
	plt.show()

def draw_plot_diff_chunks(df_ensemble,df_single): # Figure 7.b: Varying ChunkSize
	fig, ax1 = plt.subplots(figsize=(7, 6))

	ax1.plot(df_ensemble['chunk'], df_ensemble['MSE'], label="Ensemble Ek", color="#1f77b4", marker="^")
	ax1.plot(df_single['chunk'], df_single['MSE'], label="Single Gk", linestyle="--", color="#1f77b4")
	ax1.set_xticks(df_ensemble['chunk'])
	ax1.set_xlabel('ChunkSize')
	ax1.set_ylabel("Error Rate (%)")


	fig.legend()
	plt.show()



def draw_plot_diff_Ks(df_ensemble,df_single): 
	fig, ax1 = plt.subplots(figsize=(7, 6))

	ax1.plot(df_ensemble['K'], df_ensemble['MSE'], label="Ensemble Ek", color="#1f77b4", marker="^")
	ax1.plot(df_single['K'], df_single['MSE'], label="Single Gk", linestyle="--", color="#1f77b4")
	ax1.set_xticks(df_ensemble['K'])
	ax1.set_xlabel('K')
	ax1.set_ylabel("MSE")

	fig.legend()
	plt.show()





if __name__ == '__main__':

	try:
		figure = int(input('\033[91m'+'choose the figure between 1 to 5: \n 1:Figure 5 \n 2: figure 7a \n 3: figure 7b \n 4: figure 6a \n 5: figure 6b \n'+'\033[0m'))
	except Exception as ex:
		print(ex)
		print('\033[91m'+'please choose a number between 1 to 5'+'\033[0m')

	if figure == 1: #Figure 5
		K = input('Enter k (int): \n ')
		MAXSample = input('Enter MAXSample (int): \n ')
		folder_name = 'diff_chunk'
		df_ensemble = pd.read_csv(gen_filename(K,MAXSample,str(folder_name),True))
		df_single = pd.read_csv(gen_filename(K,MAXSample,str(folder_name),False))
		draw_plot_diff_chunks_time(df_ensemble,df_single)


	elif figure == 2: #figure 7a
		chunk_size = input('Enter chunk size (int): \n ')
		MAXSample = input('Enter MAXSample (int): \n ')
		folder_name = 'diff_k'
		df_ensemble = pd.read_csv(gen_filename(chunk_size,MAXSample,str(folder_name),True))
		df_single = pd.read_csv(gen_filename(chunk_size,MAXSample,str(folder_name),False))
		draw_plot_diff_Ks(df_ensemble,df_single)


	elif figure == 3:#figure 7b 
		K = input('Enter k (int): \n ')
		MAXSample = input('Enter MAXSample (int): \n ')
		folder_name = 'diff_chunk'
		df_ensemble = pd.read_csv(gen_filename(K,MAXSample,str(folder_name),True))
		df_single = pd.read_csv(gen_filename(K,MAXSample,str(folder_name),False))
		draw_plot_diff_chunks(df_ensemble,df_single)

	elif figure == 4:#figure 6a
		pass #TODO

	elif figure == 5:#figure 6b
		pass #TODO

	else:
		print('\033[91m'+'please choose a number between 1 to 5'+'\033[0m')
		sys.exit(0)
	


