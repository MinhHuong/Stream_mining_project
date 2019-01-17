import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Plot the training time in terms of K.
Each data point computes the MSE averaged over all chunk sizes
"""

def make_filename(K, chunk, max_sample):
    return "K" + str(K) + "_chunk" + str(chunk) + "_MAXSample" + str(max_sample) + ".csv"

ks = range(1, 51, 1)
chunks = range(200, 1100, 100)
max_sample = 20000
prefix = "experiments/ensemble/diff_k/"

mean_mse_k = np.zeros(len(ks))
for i, k in enumerate(ks):
    mean_mse_chunks = 0
    for chunk in chunks:
        filename = prefix + make_filename(k, chunk, max_sample)
        df = pd.read_csv(filename, sep=",", skiprows=4)
        mean_mse_chunks += np.mean(df["mean_mse_[M0]"].values)
    mean_mse_k[i] = mean_mse_chunks / len(chunks)

plt.figure(figsize=(10, 7))
plt.plot(ks, mean_mse_k)
plt.xlabel("K")
plt.ylabel("MSE")
plt.show()