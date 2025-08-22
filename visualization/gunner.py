import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import numpy as np

methods = ["CSD", "METRA", "LSD", "DIAYN", "SUSD"]

def lim(): 
    all_dfs = []
    for method in methods:
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
        reader = SummaryReader(f"./exp/HRL_{method}_nolim", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard

# def plot_result(df, save_path, title):
#     plt.figure(figsize=(10,6))

#     window = 5
#     for method, group in df.groupby("method"):
#         group_sorted = group.sort_values("step").copy()

#         # compute mean and CI for each list
#         group_sorted["mean"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
#             lambda x: np.mean(x)
#         )
#         group_sorted["ci95"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
#             lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
#         )


#         # rolling mean for smoothing
#         smoothed = group_sorted["mean"].rolling(window, min_periods=1).mean()
#         smoothed_ci = group_sorted["ci95"].rolling(window, min_periods=1).mean()

#         # plot line
#         plt.plot(
#             group_sorted["step"],
#             smoothed,
#             label=method,
#             linewidth=2,
#             alpha=0.8
#         )

#         # # plot confidence band
#         # plt.fill_between(
#         #     group_sorted["step"],
#         #     smoothed - smoothed_ci,
#         #     smoothed + smoothed_ci,
#         #     alpha=0.05
#         # )

#     # styling
#     plt.title(title, fontsize=16, weight="bold")
#     plt.xlabel("Episodes", fontsize=14)
#     plt.ylabel("Return", fontsize=14)

#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
#     plt.tight_layout()

#     # save before showing
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()


def plot_result(df, save_path, title):
    plt.figure(figsize=(10,6))

    window = 5
    for method, group in df.groupby("method"):
        group_sorted = group.sort_values("step")
        smoothed = group_sorted["EvalOp/AverageDiscountedReturn"].rolling(window, min_periods=1).mean()
        plt.plot(
            group_sorted["step"],
            smoothed,
            label=method,
            linewidth=2,
            alpha=0.6
        )

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


### gunner_lim
save_path = "visualization/vis/gunner_lim.png" 
mp_fp_diff = lim()
plot_result(mp_fp_diff, save_path, title="Gunner Limitation")


### gunner_nolim
save_path = "visualization/vis/gunner_nolim.png" 
mp_fp_hard = nolim()
plot_result(mp_fp_hard, save_path, title="Gunner No Limitation")
