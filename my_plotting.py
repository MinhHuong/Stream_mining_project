import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

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
    mpl.style.use("seaborn")

    ks = range(5, 15, 1)
    K = len(ks)
    chunks = range(200, 1100, 100)
    min_k, max_k = 5, 15

    # get time of hoeffding tree (chunk size = 200)
    ht = pd.read_csv("experiments/single/diff_k/summary/chunk200_MAXSample20000_summary.csv", sep=",")
    ht_time = ht["time"].values[min_k:max_k]

    # get time & MSE of forest by K (chunk size = 200)
    forest = pd.read_csv("experiments/rf/diff_k/summary/chunk200_MAXSample20000_summary.csv", sep=",")
    forest_time = forest["time"].values[min_k:max_k]
    forest_mse = forest["MSE"].values[min_k:max_k]

    # get time of awe by K (chunk size = 200)
    # awe = pd.read_csv("experiments/awe/chunk200_MAXSample20000_summary.csv", sep=",")
    # awe_time = awe["time"].values[min_k:max_k]
    # awe_mse = awe["MSE"].values[min_k:max_k]

    awe_mse = np.zeros(K)
    for ik, k in enumerate(ks):
        mean_mse_chunk = np.zeros(len(ks))
        for ic, chunk in enumerate(chunks):
            filename = "k" + str(k) + "_chunk" + str(chunk) + "_maxsamples200000_accukappa.csv"
            awe = pd.read_csv("experiments/awe/" + filename, sep=",", skiprows=4)
            mean_mse_chunk[ic] = np.mean(awe["current_kappa_[M0]"].values) * 100
        awe_mse[ik] = np.mean(mean_mse_chunk)

    # get K's
    # k = awe["K"].values[min_k:max_k]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    # ax1.bar(np.arange(min_k, max_k) - 0.15, forest_mse, width=0.3, label="Random Forest", color="#7f7f7f")
    ax1.plot(np.arange(min_k, max_k), forest_mse, label="Random Forest", color="#7f7f7f", marker="o")
    # ax1.plot(np.arange(min_k, max_k), awe, label="Random Forest", color="#7f7f7f", marker="o")
    # ax1.bar(np.arange(min_k, max_k) + 0.15, awe_time, width=0.3, label="AWE", color="#ff7f0e")
    ax1.set_ylabel("MSE (%)")
    ax1.set_xlabel("K")

    # double plot
    # fig, ax1 = plt.subplots(figsize=(8, 5))
    # ax1.bar(k - 0.15, forest_time, width=0.3, label="Random Forest", color="#7f7f7f")
    # ax1.bar(k + 0.15, awe_time, width=0.3, label="AWE", color="#ff7f0e")
    # ax1.set_ylabel("Training time (s)")

    # ax2 = ax1.twinx()
    # ax2.plot(k, forest_mse, linewidth=2.0, marker="^", linestyle="--", color="#2e2e2e")
    # ax2.plot(k, awe_mse, linewidth=2.0, marker="o", color="#ff7f0e")
    # ax2.set_ylabel("MSE (%)")

    plt.title("ChunkSize = 200, K varies")
    plt.xticks(ks)
    fig.legend()
    plt.show()

def plot_moa_skflow_bychunk():
    # x is chunk, y average mse by all k from 5, 10, 15
    chunks = np.arange(200, 1100, 200)
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
            filename = "k" + str(k) + "_chunk" + str(chunk) + "_maxsamples200000_accukappa.csv"
            ours = pd.read_csv("experiments/awe/" + filename, sep=",", skiprows=4)
            mean_kappa_k[j] = np.mean(ours["current_kappa_[M0]"].values) * 100
        our_kappa_chunks[i] = np.mean(mean_kappa_k)

    # Hoeffding Tree of skmultiflow
    # ht_kappa_chunks = np.zeros(len(chunks))
    # for i, chunk in enumerate(chunks):
    #     mean_kappa_k = np.zeros(len(ks))
    #     for j, k in enumerate(ks):
    #         filename = "K" + str(k) + "_chunk" + str(chunk) + "_MAXSample" + str(num_chunks * chunk) \
    #                    + "_accu_kappa.csv"
    #         ours = pd.read_csv("experiments/single_k10/diff_k/" + filename, sep=",", skiprows=5)
    #         mean_kappa_k[j] = np.mean(ours["current_kappa_[M0]"].values)
    #     ht_kappa_chunks[i] = np.mean(mean_kappa_k)

    # Hoeffding Tree by MOA
    # ht_moa_kappa_chunks = np.zeros(len(chunks))
    # for i, chunk in enumerate(chunks):
    #     mean_kappa_k = np.zeros(len(ks))
    #     for j, k in enumerate(ks):
    #         filename = "chunk" + str(chunk) + "_MAXSample" + str(num_chunks * chunk) + ".txt"
    #         moa = pd.read_csv("experiments/single_k10/diff_k/" + filename, sep=",")
    #         mean_kappa_k[j] = np.mean(moa["Kappa Statistic (percent)"].values)
    #     ht_moa_kappa_chunks[i] = np.mean(mean_kappa_k)

    plt.figure(figsize=(7, 5))
    plt.plot(chunks, our_kappa_chunks, marker="o", label="AWE (skmultiflow)")
    plt.plot(chunks, moa_kappa_chunks, marker="v", label="AWE (MOA)", linestyle="--")
    # plt.plot(chunks, ht_kappa_chunks, marker="o", label="Hoeffding Tree (skmultiflow)")
    # plt.plot(chunks, ht_moa_kappa_chunks, marker="v", label="Hoeffding Tree (MOA)", linestyle="--")
    plt.title("Comparison of MOA and scikit-multiflow implementation")
    plt.xlabel("ChunkSize")
    plt.ylim(30, 80)
    plt.ylabel("Kappa (%)")
    plt.legend()
    plt.show()

def plot_awe_kappa_acc_bychunks_moa():
    mpl.style.use("seaborn")

    # Ks = range(5, 16, 1)
    Ks = [5, 10, 15]
    chunks = range(200, 1100, 100)
    S = len(chunks)
    K = len(Ks)

    # by ours
    mean_acc_chunks = np.zeros(S)
    mean_kappa_chunks = np.zeros(S)
    for ic, chunk in enumerate(chunks):
        mean_acc_by_k = np.zeros(K)
        mean_kappa_by_k = np.zeros(K)
        for ik, k in enumerate(Ks):
            filename = "k" + str(k) + "_chunk" + str(chunk) + "_maxsamples200000_accukappa.csv"
            awe = pd.read_csv("experiments/awe/" + filename, sep=",", skiprows=4)
            mean_acc_by_k[ik] = awe["mean_acc_[M0]"].values[-1]
            mean_kappa_by_k[ik] = awe["mean_kappa_[M0]"].values[-1]
        mean_acc_chunks[ic] = np.mean(mean_acc_by_k) * 100
        mean_kappa_chunks[ic] = np.mean(mean_kappa_by_k) * 100

    # by MOA
    moa_kappa_chunks = np.zeros(S)
    moa_acc_chunks = np.zeros(S)
    for i, chunk in enumerate(chunks):
        mean_kappa_k = np.zeros(K)
        mean_acc_k = np.zeros(K)
        for j, k in enumerate(Ks):
            filename = make_filename(K=k, chunk=chunk, get_time=False, ext=".csv.txt",
                                    max_sample=100 * chunk)
            moa = pd.read_csv("experiments/moa/diff_k/" + filename, sep=",")
            mean_kappa_k[j] = moa["Kappa Statistic (percent)"].values[-1]
            mean_acc_k[j] = moa["classifications correct (percent)"].values[-1]
        moa_kappa_chunks[i] = np.mean(mean_kappa_k)
        moa_acc_chunks[i] = np.mean(mean_acc_k)

    plt.figure(figsize=(7, 5))

    plt.bar(np.arange(200, 1100, 100)-20, mean_acc_chunks, width=40, label="Accuracy (Ours)")
    plt.bar(np.arange(200, 1100, 100)+20, mean_kappa_chunks, width=40, color="#e377c2", label="Kappa (Ours)")

    plt.plot(np.arange(200, 1100, 100)-20, moa_acc_chunks, color="#17becf", label="Accuracy (MOA)", marker="o", linestyle="--")
    plt.plot(np.arange(200, 1100, 100)+20, moa_kappa_chunks, color="#d62728", label="Kappa (MOA)", marker="*", linestyle="--")

    plt.title("Accuracy and Kappa by AWE (K from 5 to 15)")
    plt.xlabel("ChunkSize")
    plt.ylabel("Accuracy/Kappa (%)")
    plt.xticks(chunks)
    plt.legend()

    plt.show()

def plot_time_by_chunks():
    """
    Plots training time by chunk size, each point is the average time by all k from 5 to 20
    :return:
    """
    mpl.style.use("seaborn")

    data = pd.read_csv("experiments/awe/expe_maxsamples200000_time.csv")
    mean_time = data.groupby(by="S")["time"].mean().reset_index()
    print(mean_time)

    chunks = mean_time["S"]
    times = mean_time["time"]

    plt.figure(figsize=(6, 4))
    plt.bar(chunks, times, width=40)
    plt.xticks(chunks)
    plt.xlabel("ChunkSize")
    plt.ylabel("Time (s)")
    plt.title("Training time by chunk size (max samples = 200,000)")
    plt.show()


def plot_time_by_ks():
    """
    Plots training time by chunk size, each point is the average time by all k from 5 to 20
    :return:
    """
    mpl.style.use("seaborn")

    data = pd.read_csv("experiments/awe/expe_maxsamples200000_time.csv")
    mean_time = data.groupby(by="k")["time"].mean().reset_index()
    print(mean_time)

    ks = mean_time["k"]
    times = mean_time["time"]

    plt.figure(figsize=(6, 4))
    plt.bar(ks, times, color="#9467bd", width=0.5)
    plt.xticks(ks)
    plt.xlabel("K")
    plt.ylabel("Time (s)")
    plt.title("Training time by #classifiers (max samples = 200,000)")
    plt.show()

def mean_by_k(cv, score, chunks, ks):
    """

    :param cv: a number 3, 5, or 10
    :param score: score = "mean_acc_[M0]" or "mean_kappa_[M0]"
    :return: mean score by ensemble size
    """
    K = len(ks)
    S = len(chunks)

    mean_score_k = np.zeros(K)
    for ik, k in enumerate(ks):
        _by_chunk = np.zeros(S)
        for ic, c in enumerate(chunks):
            filename = "k" + str(k) + "_chunk" + str(c) + "_cv" + str(cv) + "_maxsamples100000_accukappa.csv"
            data = pd.read_csv("experiments/cv/" + filename, sep=",", skiprows=4)
            _by_chunk[ic] = data[score].values[-1]
        mean_score_k[ik] = np.mean(_by_chunk)
    return mean_score_k


def mean_by_chunk(cv, score, chunks, ks):
    K = len(ks)
    S = len(chunks)

    mean_score_s = np.zeros(S)
    for ic, c in enumerate(chunks):
        _by_k = np.zeros(K)
        for ik, k in enumerate(ks):
            filename = "k" + str(k) + "_chunk" + str(c) + "_cv" + str(cv) + "_maxsamples100000_accukappa.csv"
            data = pd.read_csv("experiments/cv/" + filename, sep=",", skiprows=4)
            _by_k[ik] = data[score].values[-1]
        mean_score_s[ic] = np.mean(_by_k)
    return mean_score_s


def plot_awe_kappa_acc_by_ks():
    """
    Plots by K, average over all chunk sizes, with a given CV fold
    :param cv:
    :return:
    """
    mpl.style.use("seaborn")

    chunks = [250, 500, 750, 1000]
    S = len(chunks)

    ks = [5, 10, 15]
    K = len(ks)

    ## ACCURACY ##
    mean_acc_k_cv0 = mean_by_k(cv=0, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_k_cv3 = mean_by_k(cv=3, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_k_cv5 = mean_by_k(cv=5, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_k_cv10 = mean_by_k(cv=10, score="mean_acc_[M0]", chunks=chunks, ks=ks)

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(5, 20, 5) - 0.75, mean_acc_k_cv0, width=0.5, label="No CV")
    plt.bar(np.arange(5, 20, 5) - 0.25, mean_acc_k_cv3, width=0.5, label="3-CV")
    plt.bar(np.arange(5, 20, 5) + 0.25, mean_acc_k_cv5, width=0.5, label="5-CV")
    plt.bar(np.arange(5, 20, 5) + 0.75, mean_acc_k_cv10, width=0.5, label="10-CV")
    plt.title("Accuracy by K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.xticks([5, 10, 15])
    plt.legend()
    plt.show()

    ## KAPPA ##
    mean_kappa_k_cv0 = mean_by_k(cv=0, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_k_cv3 = mean_by_k(cv=3, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_k_cv5 = mean_by_k(cv=5, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_k_cv10 = mean_by_k(cv=10, score="mean_kappa_[M0]", chunks=chunks, ks=ks)

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(5, 20, 5) - 0.75, mean_kappa_k_cv0, width=0.5, label="No CV")
    plt.bar(np.arange(5, 20, 5) - 0.25, mean_kappa_k_cv3, width=0.5, label="3-CV")
    plt.bar(np.arange(5, 20, 5) + 0.25, mean_kappa_k_cv5, width=0.5, label="5-CV")
    plt.bar(np.arange(5, 20, 5) + 0.75, mean_kappa_k_cv10, width=0.5, label="10-CV")
    plt.title("Kappa by K")
    plt.xlabel("K")
    plt.ylabel("Kappa")
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.xticks([5, 10, 15])
    plt.legend()
    plt.show()


def plot_awe_acc_kappa_by_chunks():
    mpl.style.use("seaborn")

    chunks = [250, 500, 750, 1000]
    S = len(chunks)

    ks = [5, 10, 15]
    K = len(ks)

    ## ACCURACY ##
    mean_acc_s_cv0 = mean_by_chunk(cv=0, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_s_cv3 = mean_by_chunk(cv=3, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_s_cv5 = mean_by_chunk(cv=5, score="mean_acc_[M0]", chunks=chunks, ks=ks)
    mean_acc_s_cv10 = mean_by_chunk(cv=10, score="mean_acc_[M0]", chunks=chunks, ks=ks)

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(250, 1100, 250) - 30, mean_acc_s_cv0, width=20, label="No CV")
    plt.bar(np.arange(250, 1100, 250) - 10, mean_acc_s_cv3, width=20, label="3-CV")
    plt.bar(np.arange(250, 1100, 250) + 10, mean_acc_s_cv5, width=20, label="5-CV")
    plt.bar(np.arange(250, 1100, 250) + 30, mean_acc_s_cv10, width=20, label="10-CV")
    plt.title("Accuracy by ChunkSize")
    plt.xlabel("ChunkSize")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.xticks(chunks)
    plt.xlim(0, 1200)
    plt.legend()
    plt.show()

    ## ACCURACY ##
    mean_kappa_s_cv0 = mean_by_chunk(cv=0, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_s_cv3 = mean_by_chunk(cv=3, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_s_cv5 = mean_by_chunk(cv=5, score="mean_kappa_[M0]", chunks=chunks, ks=ks)
    mean_kappa_s_cv10 = mean_by_chunk(cv=10, score="mean_kappa_[M0]", chunks=chunks, ks=ks)

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(250, 1100, 250) - 30, mean_kappa_s_cv0, width=20, label="No CV")
    plt.bar(np.arange(250, 1100, 250) - 10, mean_kappa_s_cv3, width=20, label="3-CV")
    plt.bar(np.arange(250, 1100, 250) + 10, mean_kappa_s_cv5, width=20, label="5-CV")
    plt.bar(np.arange(250, 1100, 250) + 30, mean_kappa_s_cv10, width=20, label="10-CV")
    plt.title("Kappa by ChunkSize")
    plt.xlabel("ChunkSize")
    plt.ylabel("Kappa")
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.xticks(chunks)
    plt.xlim(0, 1200)
    plt.legend()
    plt.show()


def plot_time_by_k_chunks_cv():
    """
    Plots training time by chunk size, each point is the average time by all k from 5 to 20
    :return:
    """
    mpl.style.use("seaborn")
    ks = [5, 10, 15]
    chunks = [250, 500, 750, 1000]

    data = pd.read_csv("experiments/cv/expe_cv_maxsamples100000_time.csv")

    mean_time = data.groupby(by=["cv", "k"])["time"].mean().reset_index()
    times_by_k_cv3 = mean_time[mean_time.cv == 3]["time"]
    times_by_k_cv5 = mean_time[mean_time.cv == 5]["time"]
    times_by_k_cv10 = mean_time[mean_time.cv == 10]["time"]

    plt.figure(figsize=(6, 4))
    plt.plot(ks, times_by_k_cv3, marker="o", label="3-fold CV")
    plt.plot(ks, times_by_k_cv5, marker="o", label="5-fold CV")
    plt.plot(ks, times_by_k_cv10, marker="o", label="10-fold CV")
    plt.xticks(np.arange(5, 16, 2))
    plt.xlabel("K")
    plt.ylabel("Time (s)")
    plt.title("Training time by #classifiers (max samples = 100,000)")
    plt.legend()
    plt.ylim((100, 300))
    plt.show()

    mean_time = data.groupby(by=["cv", "S"])["time"].mean().reset_index()

    times_by_s_cv3 = mean_time[mean_time.cv == 3]["time"]
    times_by_s_cv5 = mean_time[mean_time.cv == 5]["time"]
    times_by_s_cv10 = mean_time[mean_time.cv == 10]["time"]

    plt.figure(figsize=(6, 4))
    plt.plot(chunks, times_by_s_cv3, marker="o", label="3-fold CV")
    plt.plot(chunks, times_by_s_cv5, marker="o", label="5-fold CV")
    plt.plot(chunks, times_by_s_cv10, marker="o", label="10-fold CV")
    plt.xticks(chunks)
    plt.xlabel("ChunkSize")
    plt.ylabel("Time (s)")
    plt.title("Training time by ChunkSize (max samples = 100,000)")
    plt.legend()
    plt.ylim((100, 300))
    plt.show()


# plot_time_awe_forest()
# plot_time_hoeffding_awe()
# plot_moa_skflow_bychunk()
# plot_awe_kappa_acc_bychunks()
# plot_time_by_chunks()
# plot_time_by_ks()
# plot_awe_acc_kappa_by_chunks()
plot_time_by_k_chunks_cv()