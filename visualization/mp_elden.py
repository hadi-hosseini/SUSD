import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import numpy as np
import math

# methods = ["CSD", "METRA", "LSD", "DIAYN", "SUSD", "DUSDI"]
# methods = ["ABLATION", "SUSD"]
methods = ["SUSD (IND)", "SUSD"]


def fp_diff(): # task_diff = 4
    all_dfs = []
    for method in methods:
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_4_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_3_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_2_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_fp_1_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_5_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_6_ABLATION2", pivot=True)
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
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_seq_7_ABLATION2", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_seq_7", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard


def elden_BiP(): 
    all_dfs = []
    for method in methods:
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_elden_BiP_ABLATION", pivot=True)
        elif method == "SUSD (IND)":
            reader = SummaryReader(f"./exp/HRL_SUSD_elden_BiP_IND", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_elden_BiP", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        # df = df[df["step"] <= 5000]
        all_dfs.append(df)

    bip = pd.concat(all_dfs)
    return bip

def elden_MiP():
    all_dfs = []
    for method in methods:
        if method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_elden_MiP_ABLATION", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_elden_MiP", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        # df = df[df["step"] <= 5000]
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
        # df = df[df["step"] <= 5000]
        all_dfs.append(df)

    pos = pd.concat(all_dfs)
    return pos




def lim(): 
    all_dfs = []
    for method in methods:
        if method == "ABLATION1":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_ABLATION1", pivot=True)
        elif method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_ABLATION2", pivot=True)
        elif method == "DUSDI":
            reader = SummaryReader(f"./exp/HRL_{method}_lim_V2", pivot=True)
        elif method == "SUSD (D=1)":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_DIM_1", pivot=True)
        elif method == "SUSD (D=2)" or method == "SUSD Continuous":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim", pivot=True)
        elif method == "SUSD Discrete":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_DISC", pivot=True)     
            # reader = SummaryReader(f"./exp/HRL_SUSD_lim_dusdi", pivot=True)                   
        elif method == "SUSD (D=5)":
            reader = SummaryReader(f"./exp/HRL_SUSD_lim_dim_5", pivot=True)
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
        elif method == "Ablation":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_ABLATION2", pivot=True)
        elif method == "DUSDI":
            reader = SummaryReader(f"./exp/HRL_{method}_nolim_V2", pivot=True)
        elif method == "SUSD (N=20)":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_V2", pivot=True)
        elif method == "SUSD (D=1)":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_DIM_1", pivot=True)
        elif method == "SUSD (D=2)" or method == "SUSD Continuous":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim", pivot=True)
        elif method == "SUSD Discrete":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_DISC", pivot=True)    
            # reader = SummaryReader(f"./exp/HRL_SUSD_n`olim_dusdi", pivot=True)    
        elif method == "SUSD (D=5)":
            reader = SummaryReader(f"./exp/HRL_SUSD_nolim_dim_5", pivot=True)
        else:
            reader = SummaryReader(f"./exp/HRL_{method}_nolim", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_hard = pd.concat(all_dfs)
    return seq_hard


# def elden_PoT():
#     all_dfs = []
#     for method in methods:
#         reader = SummaryReader(f"./exp/HRL_{method}_elden_PoT", pivot=True)
#         df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
#         df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
#         df["method"] = method
#         df = df[df["step"] <= 5000]
#         all_dfs.append(df)

#     pot = pd.concat(all_dfs)
#     return pot


def plot_result(df, save_path, title):
    plt.figure(figsize=(10,6))
    plot_result_on_ax(df, plt.gca(), title)
    plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_result_on_ax(df, ax, title, window=10, has_margin=True):
    for method, group in df.groupby("method"):
        group_sorted = group.sort_values("step").copy()

        # compute mean and CI
        group_sorted["mean"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(lambda x: np.mean(x))
        if has_margin:
            group_sorted["ci95"] = group_sorted["EvalOp/AverageDiscountedReturn"].apply(
                lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
            )

        smoothed = group_sorted["mean"].rolling(window, min_periods=1).mean()
        if has_margin:
            smoothed_ci = group_sorted["ci95"].rolling(window, min_periods=1).mean()

        ax.plot(group_sorted["step"], smoothed, label=method, linewidth=2, alpha=0.8)

        max_margin = 3
        if has_margin:
            ci_clipped = np.minimum(smoothed_ci, max_margin)
        else:
            ci_clipped = 0
        ax.fill_between(group_sorted["step"], smoothed - ci_clipped, smoothed + ci_clipped, alpha=0.05)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_grouped_results(dfs, titles, save_path=None, ncols=3, figsize=(15, 8)):
    nrows = math.ceil(len(dfs) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0], nrows * figsize[1] / 2))
    axs = axs.flatten()

    for i, (df, title) in enumerate(zip(dfs, titles)):
        if "Multi" in title:
            plot_result_on_ax(df, axs[i], title, window=10)
        elif "Kitchen" in title:
            plot_result_on_ax(df, axs[i], title, window=500, has_margin=True)
        elif "Gunner" in title:
            plot_result_on_ax(df, axs[i], title, window=10)

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



# ### mp_fp_diff
# save_path = "visualization/vis/mp_fp_diff.png" 
# mp_fp_diff = fp_diff()
# # plot_result(mp_fp_diff, save_path, title="Multiparticle Food&Poison Difficult")


# ### mp_fp_hard
# save_path = "visualization/vis/mp_fp_hard.png" 
# mp_fp_hard = fp_hard()
# # plot_result(mp_fp_hard, save_path, title="Multiparticle Food&Poison Hard")


# ### mp_fp_medium
# save_path = "visualization/vis/mp_fp_medium.png" 
# mp_fp_medium = fp_medium()
# # plot_result(mp_fp_medium, save_path, title="Multiparticle Food&Poison Medium")

# ### mp_fp_easy
# save_path = "visualization/vis/mp_fp_easy.png" 
# mp_fp_easy = fp_easy()
# # plot_result(mp_fp_easy, save_path, title="Multiparticle Food&Poison Easy")

# ### mp seq_easy
# save_path = "visualization/vis/mp_seq_easy.png" 
# mp_seq_easy = seq_easy()
# # plot_result(mp_seq_easy, save_path, title="Multiparticle Sequential Easy")

# ### mp seq_medium
# save_path = "visualization/vis/mp_seq_medium.png" 
# mp_seq_medium = seq_medium()
# # plot_result(mp_seq_medium, save_path, title="Multiparticle Sequential Medium")

# ### mp seq_hard
# save_path = "visualization/vis/mp_seq_hard.png" 
# mp_seq_hard = seq_hard()
# # plot_result(mp_seq_hard, save_path, title="Multiparticle Sequential Hard")


### elden_BiP
save_path = "visualization/vis/elden_BiP.png" 
bip = elden_BiP()
# plot_result(bip, save_path, title="Put Butter in Pot Task")


# ### elden_Mip
# save_path = "visualization/vis/elden_MiP.png" 
# mip = elden_MiP()
# # plot_result(mip, save_path, title="Put Meatball in Pot Task")


# ### gunner_lim
# save_path = "visualization/ablation/gunner_lim.png" 
# gunner_lim = lim()
# # plot_result(gunner_lim, save_path, title="Gunner Limitation")


# ## gunner_nolim
# save_path = "visualization/ablation/gunner_nolim.png" 
# gunner_nolim = nolim()
# # plot_result(gunner_nolim, save_path, title="Gunner No Limitation")


# ### elden_PoS
# save_path = "visualization/vis/elden_PoS.png" 
# pos = elden_PoS()
# # plot_result(pos, save_path, title="Put Pot on Stove Task")


### elden_PoT
# save_path = "visualization/vis/elden_PoT.png" 
# pot = elden_PoT()
# # plot_result(pot, save_path, title="Put Pot in Target Task")


### plot_groups
# dfs = [mp_fp_diff, mp_fp_hard, mp_fp_medium, mp_fp_easy, mp_seq_easy, mp_seq_medium, mp_seq_hard, gunner_lim, gunner_nolim, bip, mip, pos]
# titles = [
#     "Multiparticle Food&Poison Difficult",
#     "Multiparticle Food&Poison Hard",
#     "Multiparticle Food&Poison Medium",
#     "Multiparticle Food&Poison Easy",
#     "Multiparticle Sequential Easy",
#     "Multiparticle Sequential Medium",
#     "Multiparticle Sequential Hard",
#     "Gunner with Limitation",
#     "Gunner without Limitation",
#     "Kitchen Butter in Pot",
#     "Kitchen Meatball in Pot",
#     "Kitchen Pot on Stove",
# ]

# # save_path = "visualization/vis/ablations_grouped_results.png"
# save_path = "visualization/vis/grouped_results.png"
# plot_grouped_results(dfs, titles, save_path=save_path, ncols=3)




### plot_groups
dfs = [bip]
titles = [
    "Kitchen Butter in Pot",
    "Kitchen Meatball in Pot",
]

save_path = "visualization/vis/factor_0_grouped_results.png"
plot_grouped_results(dfs, titles, save_path=save_path, ncols=2)