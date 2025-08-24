import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import numpy as np

methods = ["CSD", "ABLATION1", "SUSD"]

def fp_diff(): # task_diff = 4
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_4_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_fp_4", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_diff = pd.concat(all_dfs)
    return fp_diff

def fp_hard(): # task_diff = 3
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_3_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_fp_3", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_hard = pd.concat(all_dfs)
    return fp_hard

def fp_medium(): # task_diff = 2
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_2_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_fp_2", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_medium = pd.concat(all_dfs)
    return fp_medium

def fp_easy(): # task_diff = 1
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_1_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_fp_1", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_easy = pd.concat(all_dfs)
    return fp_easy

def seq_easy(): # task_diff = 5
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_5_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_seq_5", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_easy = pd.concat(all_dfs)
    return seq_easy

def seq_medium(): # task_diff = 6
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_6_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_seq_6", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_medium = pd.concat(all_dfs)
    return seq_medium

def seq_hard(): # task_diff = 7
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_7_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_seq_7", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard


def lim(): 
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_lim", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard

def nolim():
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_ABLATION1", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_nolim", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard

def plot_result(df, save_path, title):
    plt.figure(figsize=(10,6))

    window = 5
    for method, group in df.groupby("method"):
        group_sorted = group.sort_values("step").copy()

        # compute mean and CI for each list
        group_sorted["mean"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
            lambda x: np.mean(x)
        )
        # group_sorted["ci95"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
        #     lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
        # )


        # rolling mean for smoothing
        smoothed = group_sorted["mean"].rolling(window, min_periods=1).mean()
        # smoothed_ci = group_sorted["ci95"].rolling(window, min_periods=1).mean()

        # plot line
        plt.plot(
            group_sorted["step"],
            smoothed,
            label=method,
            linewidth=2,
            alpha=0.8
        )

        # # plot confidence band
        # plt.fill_between(
        #     group_sorted["step"],
        #     smoothed - smoothed_ci,
        #     smoothed + smoothed_ci,
        #     alpha=0.05
        # )

    # styling
    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Return", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
    plt.tight_layout()

    # save before showing
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# def plot_result(df, save_path, title):
#     plt.figure(figsize=(10,6))

#     window = 5
#     for method, group in df.groupby("method"):
#         group_sorted = group.sort_values("step")
#         smoothed = group_sorted["EvalOp/AverageDiscountedReturn"].rolling(window, min_periods=1).mean()
#         plt.plot(
#             group_sorted["step"],
#             smoothed,
#             label=method,
#             linewidth=2,
#             alpha=0.6
#         )

#     # styling
#     plt.title(title, fontsize=16, weight="bold")
#     plt.xlabel("Steps", fontsize=14)
#     plt.ylabel("Return", fontsize=14)

#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
#     plt.tight_layout()

#     # save before showing
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

### mp_fp_diff
save_path = "visualization/ablation/mp_fp_diff.png" 
mp_fp_diff = fp_diff()
plot_result(mp_fp_diff, save_path, title="Multiparticle Food&Poison Difficult")


### mp_fp_hard
save_path = "visualization/ablation/mp_fp_hard.png" 
mp_fp_hard = fp_hard()
plot_result(mp_fp_hard, save_path, title="Multiparticle Food&Poison Hard")


### mp_fp_medium
save_path = "visualization/ablation/mp_fp_medium.png" 
mp_fp_medium = fp_medium()
plot_result(mp_fp_medium, save_path, title="Multiparticle Food&Poison Medium")

### mp_fp_easy
save_path = "visualization/ablation/mp_fp_easy.png" 
mp_fp_easy = fp_easy()
plot_result(mp_fp_easy, save_path, title="Multiparticle Food&Poison Easy")


### gunner_lim
save_path = "visualization/ablation/gunner_lim.png" 
mp_fp_diff = lim()
plot_result(mp_fp_diff, save_path, title="Gunner Limitation")


### gunner_nolim
save_path = "visualization/ablation/gunner_nolim.png" 
mp_fp_hard = nolim()
plot_result(mp_fp_hard, save_path, title="Gunner No Limitation")

### mp seq_easy
save_path = "visualization/ablation/mp_seq_easy.png" 
mp_seq_easy = seq_easy()
plot_result(mp_seq_easy, save_path, title="Multiparticle Sequential Easy")

### mp seq_medium
save_path = "visualization/ablation/mp_seq_medium.png" 
mp_seq_medium = seq_medium()
plot_result(mp_seq_medium, save_path, title="Multiparticle Sequential Medium")

### mp seq_hard
save_path = "visualization/ablation/mp_seq_hard.png" 
mp_seq_hard = seq_hard()
plot_result(mp_seq_hard, save_path, title="Multiparticle Sequential Hard")