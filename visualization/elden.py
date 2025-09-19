import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import numpy as np
import math

# downstream_task = "elden_BiP" ['BiP', 'MiP', 'PoS', 'BiP_PoS', 'MiP_PoS', 'PoT']
methods = ["CSD", "METRA", "LSD", "DIAYN", "SUSD", "DUSDI"]

def elden_BiP(): 
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_elden_BiP", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    bip = pd.concat(all_dfs)
    return bip

def elden_MiP():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_elden_MiP", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    mip = pd.concat(all_dfs)
    return mip


def elden_PoS():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_elden_PoS", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    pos = pd.concat(all_dfs)
    return pos



def elden_PoT():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_elden_PoT", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    pos = pd.concat(all_dfs)
    return pos

def plot_result(df, save_path, title):
    plt.figure(figsize=(10,6))
    plot_result_on_ax(df, plt.gca(), title)
    plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_result_on_ax(df, ax, title):
    window = 100 # 5
    for method, group in df.groupby("method"):
        group_sorted = group.sort_values("step").copy()

        # compute mean and CI
        group_sorted["mean"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(lambda x: np.mean(x))
        # group_sorted["ci95"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
        #     lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
        # )

        smoothed = group_sorted["mean"].rolling(window, min_periods=1).mean()
        # smoothed_ci = group_sorted["ci95"].rolling(window, min_periods=1).mean()

        ax.plot(group_sorted["step"], smoothed, label=method, linewidth=2, alpha=0.8)

        max_margin = 3
        # ci_clipped = np.minimum(smoothed_ci, max_margin)
        # ax.fill_between(group_sorted["step"], smoothed - ci_clipped, smoothed + ci_clipped, alpha=0.05)
        ax.fill_between(group_sorted["step"], smoothed, smoothed, alpha=0.05)


    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_grouped_results(dfs, titles, save_path=None, ncols=3, figsize=(15, 8)):
    nrows = math.ceil(len(dfs) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0], nrows * figsize[1] / 2))
    axs = axs.flatten()

    for i, (df, title) in enumerate(zip(dfs, titles)):
        plot_result_on_ax(df, axs[i], title)

    # hide extra axes
    for j in range(len(dfs), len(axs)):
        axs[j].axis("off")

    # shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Method", fontsize=12, title_fontsize=13,
        loc="upper center", ncol=len(labels)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()




### elden_BiP
save_path = "visualization/vis/elden_BiP.png" 
bip = elden_BiP()
plot_result(bip, save_path, title="Put Butter in Pot Task")


### elden_Mip
save_path = "visualization/vis/elden_MiP.png" 
mip = elden_MiP()
plot_result(mip, save_path, title="Put Meatball in Pot Task")


### elden_PoS
save_path = "visualization/vis/elden_PoS.png" 
pos = elden_PoS()
plot_result(pos, save_path, title="Put Meatball in Pot Task")


# ### elden_PoT
# save_path = "visualization/vis/elden_PoT.png" 
# pot = elden_PoT()
# plot_result(pot, save_path, title="Put Pot in Target Task")


### plot_groups
dfs = [bip, mip, pos]
titles = [
    "Put Butter in Pot Task",
    "Put Meatball in Pot Task",
    "Put Pot on Stove",
]

save_path = "visualization/vis/elden_grouped_results.png"
plot_grouped_results(dfs, titles, save_path=save_path, ncols=3)