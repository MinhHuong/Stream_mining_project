import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Plot the training time in terms of K.
"""

def make_filename(K, chunk, max_sample, get_time=False, ext=".csv"):
    if not get_time:
        return "K" + str(K) + "_chunk" + str(chunk) + "_MAXSample" + str(max_sample) + ext
    return "summary/chunk" + str(chunk) + "_MAXSample" + str(max_sample) + "_summary" + ext

def get_mean_times(prefix):
    mean_times = np.zeros(len(chunks))

    for i, chunk in enumerate(chunks):
        max_sample = chunk * num_chunks
        filename = prefix + make_filename(chunk=chunk, K=None, max_sample=max_sample, get_time=True)
        df = pd.read_csv(filename, sep=",")
        mean_times[i] = np.mean(df["time"])

    return mean_times

def get_mean_mse(prefix):
    mean_mse_k = np.zeros(len(chunks))
    for i, k in enumerate(ks):
        mean_mse_chunks = 0
        for chunk in chunks:
            max_sample = chunk * num_chunks
            filename = prefix + make_filename(k, chunk, max_sample, False)
            df = pd.read_csv(filename, sep=",", skiprows=4)
            mean_mse_chunks += np.mean(df["mean_mse_[M0]"].values)
        mean_mse_k[i] = mean_mse_chunks / len(chunks)
    return mean_mse_k

def get_mean_kappa_moa_chunks(prefix):
    mean_kappa = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        max_sample = 100000
        filename = prefix + make_filename(K=10, chunk=chunk, max_sample=max_sample, get_time=False, ext=".txt")
        df = pd.read_csv(filename, sep=",")
        mean_kappa[i] += np.mean(df["Kappa Statistic (percent)"].values)
    return mean_kappa

ks = range(5, 51, 5)
chunks = np.arange(200, 1100, 100)
num_chunks = 100

### Plot MSE in terms of K ###
# mean_mse_k = np.zeros(len(ks))
# for i, k in enumerate(ks):
#     mean_mse_chunks = 0
#     for chunk in chunks:
#         filename = prefix + make_filename(k, chunk, max_sample, False)
#         df = pd.read_csv(filename, sep=",", skiprows=4)
#         mean_mse_chunks += np.mean(df["mean_mse_[M0]"].values)
#     mean_mse_k[i] = mean_mse_chunks / len(chunks)
#
# plt.figure(figsize=(10, 7))
# plt.plot(ks, mean_mse_k)
# plt.xlabel("K")
# plt.ylabel("MSE")
# plt.show()

### Plot training time in terms of K ###
# mean_times_single = get_mean_times("experiments/single/diff_k/")
# mean_times_ensemble = get_mean_times("experiments/ensemble/diff_k/")

# mean_mse_single = get_mean_mse("experiments/single/diff_k/")
# mean_mse_ensemble = get_mean_mse("experiments/ensemble/diff_k/")

mean_kappa_moa = get_mean_kappa_moa_chunks("experiments/moa/diff_k/")

# for i, chunk in enumerate(chunks):
#     max_sample = chunk * num_chunks
#     filename = prefix + make_filename(chunk=chunk, K=None, max_sample=max_sample, get_time=True)
#     df = pd.read_csv(filename, sep=",")
#     mean_times[i] = np.mean(df["time"])

plt.figure(figsize=(7, 5))

plt.plot(chunks, mean_kappa_moa)

# plt.plot(chunks, mean_times_single, marker='o', label="Hoeffding Tree")
# plt.plot(chunks, mean_times_ensemble, marker='o', label="Ensemble")
# plt.bar(chunks-15, mean_times_single, width=30, label="Hoeffding Tree")
# plt.bar(chunks+15, mean_times_ensemble, width=30, label="Ensemble")

# plt.plot(chunks, mean_mse_single, marker="o", label="Hoeffding Tree")
# plt.plot(mean_mse_ensemble, marker="o", label="Ensemble")

plt.xticks(chunks)
plt.xlabel("ChunkSize")
plt.ylabel("Kappa (%)")
# plt.ylabel("Training time(s)")
plt.legend()
plt.show()