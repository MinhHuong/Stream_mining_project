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

def get_mean_times(prefix, ks, chunks, num_chunks):
    mean_times = np.zeros(len(chunks))

    for i, chunk in enumerate(chunks):
        max_sample = chunk * num_chunks
        filename = prefix + make_filename(chunk=chunk, K=None, max_sample=max_sample, get_time=True)
        try:
            df = pd.read_csv(filename, sep=",")
            mean_times[i] = np.mean(df["time"])
        except FileNotFoundError:
            mean_times[i] = 0

    return mean_times

def get_mean_mse(prefix, ks, chunks, num_chunks):
    mean_mse_k = np.zeros(len(chunks))
    for i, k in enumerate(ks):
        mean_mse_chunks = 0
        for chunk in chunks:
            max_sample = chunk * k
            filename = prefix + make_filename(k, chunk, max_sample, False)
            df = pd.read_csv(filename, sep=",", skiprows=4)
            mean_mse_chunks += np.mean(df["mean_mse_[M0]"].values)
        mean_mse_k[i] = mean_mse_chunks / len(chunks)
    return mean_mse_k

def get_mean_kappa_moa_chunks(prefix, chunks):
    mean_kappa = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        max_sample = 100000
        filename = prefix + make_filename(K=10, chunk=chunk, max_sample=max_sample, get_time=False, ext=".txt")
        df = pd.read_csv(filename, sep=",")
        mean_kappa[i] += np.mean(df["Kappa Statistic (percent)"].values)
    return mean_kappa

def plot_time_hoeffding_awe():
    ks = range(5, 51, 5)
    chunks = np.arange(200, 1100, 100)
    num_chunks = 100

    mean_times_single = get_mean_times("experiments/single/diff_k/", ks, chunks, num_chunks)
    mean_times_ensemble = get_mean_times("experiments/ensemble/diff_k/", ks, chunks, num_chunks)

    plt.figure(figsize=(7, 5))
    plt.plot(chunks, mean_times_single, marker='o', label="Hoeffding Tree")
    plt.plot(chunks, mean_times_ensemble, marker='^', label="Ensemble")

    plt.xticks(chunks)
    plt.xlabel("ChunkSize")
    plt.ylabel("Training time(s)")
    plt.title("ChunkSize varies on averaged MSE of K = 1 to 50")
    plt.legend()
    plt.show()

def plot_time_awe_forest():
    ks = range(1, 51, 1)
    min_k, max_k = 4, 20

    # get time of hoeffding tree (chunk size = 200)
    ht = pd.read_csv("experiments/single/diff_k/summary/chunk200_MAXSample20000_summary.csv", sep=",")
    ht_time = ht["time"].values[min_k:max_k]

    # get time & MSE of forest by K (chunk size = 200)
    forest = pd.read_csv("experiments/rf/diff_k/summary/chunk200_MAXSample20000_summary.csv", sep=",")
    forest_time = forest["time"].values[min_k:max_k]
    forest_mse = forest["MSE"].values[min_k:max_k]

    # get time of awe by K (chunk size = 200)
    awe = pd.read_csv("experiments/ensemble/diff_k/summary/chunk200_MAXSample20000_summary.csv", sep=",")
    awe_time = awe["time"].values[min_k:max_k]
    awe_mse = awe["MSE"].values[min_k:max_k]

    # get K's
    k = awe["K"].values[min_k:max_k]

    # double plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(k - 0.15, forest_time, width=0.3, label="Random Forest", color="#7f7f7f")
    ax1.bar(k + 0.15, awe_time, width=0.3, label="AWE", color="#ff7f0e")
    ax1.set_ylabel("Training time (s)")

    ax2 = ax1.twinx()
    ax2.plot(k, forest_mse, linewidth=2.0, marker="^", linestyle="--", color="#2e2e2e")
    ax2.plot(k, awe_mse, linewidth=2.0, marker="o", color="#ff7f0e")
    ax2.set_ylabel("MSE (%)")

    plt.title("ChunkSize = 200, K varies")
    plt.xticks(k)
    fig.legend()
    plt.show()

def plot_moa_skflow_bychunk():
    # x is chunk, y average mse by all k from 5, 10, 15
    chunks = np.arange(200, 1100, 100)
    ks = np.array([5, 10, 15])
    num_chunks = 100

    # AWE of MOA
    moa_kappa_chunks = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        mean_kappa_k = np.zeros(len(ks))
        for j, k in enumerate(ks):
            filename = make_filename(K=k, chunk=chunk, get_time=False, ext=".csv.txt", max_sample=num_chunks * chunk)
            moa = pd.read_csv("experiments/moa/diff_k/" + filename, sep=",")
            mean_kappa_k[j] = np.mean(moa["Kappa Statistic (percent)"].values)
        moa_kappa_chunks[i] = np.mean(mean_kappa_k)

    # AWE of ours
    our_kappa_chunks = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        mean_kappa_k = np.zeros(len(ks))
        for j, k in enumerate(ks):
            filename = "K" + str(k) + "_chunk" + str(chunk) + "_MAXSample" + str(num_chunks * chunk) \
                       + "_accu_kappa.csv"
            ours = pd.read_csv("experiments/ensemble_k10/diff_k/" + filename, sep=",", skiprows=4)
            mean_kappa_k[j] = np.mean(ours["mean_kappa_[M0]"].values)
        our_kappa_chunks[i] = np.mean(mean_kappa_k)

    # Hoeffding Tree of skmultiflow
    ht_kappa_chunks = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        mean_kappa_k = np.zeros(len(ks))
        for j, k in enumerate(ks):
            filename = "K" + str(k) + "_chunk" + str(chunk) + "_MAXSample" + str(num_chunks * chunk) \
                       + "_accu_kappa.csv"
            ours = pd.read_csv("experiments/single_k10/diff_k/" + filename, sep=",", skiprows=5)
            mean_kappa_k[j] = np.mean(ours["mean_kappa_[M0]"].values)
        ht_kappa_chunks[i] = np.mean(mean_kappa_k)

    # Hoeffding Tree by MOA
    ht_moa_kappa_chunks = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        mean_kappa_k = np.zeros(len(ks))
        for j, k in enumerate(ks):
            filename = "chunk" + str(chunk) + "_MAXSample" + str(num_chunks * chunk) + ".txt"
            moa = pd.read_csv("experiments/single_k10/diff_k/" + filename, sep=",")
            mean_kappa_k[j] = np.mean(moa["Kappa Statistic (percent)"].values)
        ht_moa_kappa_chunks[i] = np.mean(mean_kappa_k)

    plt.figure(figsize=(7, 5))
    plt.plot(chunks, our_kappa_chunks, marker="o", label="AWE (skmultiflow)")
    plt.plot(chunks, ht_kappa_chunks, marker="o", label="Hoeffding Tree (skmultiflow)")
    plt.plot(chunks, moa_kappa_chunks, marker="v", label="AWE (MOA)", linestyle="--")
    plt.plot(chunks, ht_moa_kappa_chunks, marker="v", label="Hoeffding Tree (MOA)", linestyle="--")
    plt.title("Comparison of MOA and scikit-multiflow implementation")
    plt.xlabel("ChunkSize")
    plt.ylabel("Kappa statistics (%)")
    plt.legend()
    plt.show()

    # plt.figure(figsize=(7, 5))
    # plt.legend()
    # plt.show()

# plot_time_awe_forest()
# plot_time_hoeffding_awe()
plot_moa_skflow_bychunk()
