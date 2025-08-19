import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

methods = ["CSD", "METRA", "LSD", "DIAYN", "SUSD"]

def fp_diff():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_diff = pd.concat(all_dfs)
    return fp_diff

def fp_hard():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_3", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_hard = pd.concat(all_dfs)
    return fp_hard

def fp_medium():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_2", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_medium = pd.concat(all_dfs)
    return fp_medium



def fp_easy():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_1", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    fp_easy = pd.concat(all_dfs)
    return fp_easy


def seq_easy():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_7_seq", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_easy = pd.concat(all_dfs)
    return seq_easy


def seq_medium():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_5_seq", pivot=True)
        df = reader.scalars[["step", "EvalOp/AverageDiscountedReturn"]].copy()
        df = df.dropna(subset=["EvalOp/AverageDiscountedReturn"])
        df["method"] = method
        all_dfs.append(df)

    seq_medium = pd.concat(all_dfs)
    return seq_medium


def seq_hard():
    all_dfs = []
    for method in methods:
        reader = SummaryReader(f"./exp/HRL_{method}_6_seq", pivot=True)
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
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Return", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Method", fontsize=12, title_fontsize=13, loc="best")
    plt.tight_layout()

    # save before showing
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

### mp_fp_diff
save_path = "visualization/vis/mp_fp_diff.png" 
mp_fp_diff = fp_diff()
plot_result(mp_fp_diff, save_path, title="Multiparticle Food&Poison Difficult")


### mp_fp_hard
save_path = "visualization/vis/mp_fp_hard.png" 
mp_fp_hard = fp_hard()
plot_result(mp_fp_hard, save_path, title="Multiparticle Food&Poison Hard")


### mp_fp_hard
save_path = "visualization/vis/mp_fp_medium.png" 
mp_fp_medium = fp_medium()
plot_result(mp_fp_medium, save_path, title="Multiparticle Food&Poison Medium")

### mp_fp_easy
save_path = "visualization/vis/mp_fp_easy.png" 
mp_fp_easy = fp_easy()
plot_result(mp_fp_easy, save_path, title="Multiparticle Food&Poison Easy")


### mp seq_easy
save_path = "visualization/vis/mp_seq_easy.png" 
mp_seq_easy = seq_easy()
plot_result(mp_seq_easy, save_path, title="Multiparticle Sequential Easy")

### mp seq_medium
save_path = "visualization/vis/mp_seq_medium.png" 
mp_seq_medium = seq_medium()
plot_result(mp_seq_medium, save_path, title="Multiparticle Sequential Medium")

### mp seq_hard
save_path = "visualization/vis/mp_seq_hard.png" 
mp_seq_hard = seq_hard()
plot_result(mp_seq_hard, save_path, title="Multiparticle Sequential Hard")