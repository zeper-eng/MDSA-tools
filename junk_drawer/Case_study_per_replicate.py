from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorcet as cc
from scipy.stats import f_oneway, levene, shapiro

#################
# Load outputs
#################

BASE = Path(__file__).resolve().parent / "Outputs"
OUT = BASE / "SETD8_WT_A296T_post750ns"
HBOND_DIR = BASE / "SETD8" / "hbond_arrays"
PREFIX = "SETD8_WT_A296T"

X = np.load(OUT / f"{PREFIX}_feature_matrix.npy", mmap_mode="r")
P = np.load(OUT / f"{PREFIX}_PCA.npy")
U = np.load(OUT / f"{PREFIX}_UMAP.npy")
weights = np.load(OUT / f"{PREFIX}_PCA_weights.npy")
var = np.load(OUT / f"{PREFIX}_PCA_variance.npy")
meta = pd.read_csv(OUT / f"{PREFIX}_frame_results.csv")

SYSTEMS = ["WT", "A296T"]
REPLICATES = range(30)
meta["replicate_label"] = meta["system"] + "_clone" + meta["replicate"].astype(str)

# Only the first cached H-bond array is needed for the residue-pair lookup.
# The reduction script already checks consistent residue ordering across all 60 arrays.
A0 = np.load(HBOND_DIR / "WT_clone0.npy", mmap_mode="r")

assert len(meta) == len(X) == len(P) == len(U), "Frame counts differ between outputs"

print("Feature matrix:", X.shape)
print("PCA:", P.shape)
print("UMAP:", U.shape)
print("PCA weights:", weights.shape)
print("Frames:", len(meta))
print(meta.groupby(["system", "replicate"]).size())

#################
# Color mappings
#################

palette = cc.glasbey[:60]

def plot_system(Z, xlabel, ylabel, outfile):
    plt.figure()
    for system in SYSTEMS:
        idx = meta["system"].eq(system).to_numpy()
        plt.scatter(Z[idx, 0], Z[idx, 1], s=5, label=system)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / outfile, dpi=300)
    plt.close()

def plot_replicate(Z, xlabel, ylabel, outfile):
    plt.figure(figsize=(11, 7))
    for color, (label, group) in zip(palette, meta.groupby("replicate_label", sort=False)):
        idx = group.index.to_numpy()
        plt.scatter(Z[idx, 0], Z[idx, 1], s=5, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=6, ncol=3, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUT / outfile, dpi=300)
    plt.close()

def plot_time(Z, xlabel, ylabel, outfile):
    plt.figure()
    s = plt.scatter(Z[:, 0], Z[:, 1], c=meta["time_ns"], s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(s, label="Simulation time (ns)")
    plt.tight_layout()
    plt.savefig(OUT / outfile, dpi=300)
    plt.close()

pc1 = f"PC1 ({var[0] * 100:.1f}%)"
pc2 = f"PC2 ({var[1] * 100:.1f}%)"

plot_system(P, pc1, pc2, "PCA_by_system.png")
plot_replicate(P, pc1, pc2, "PCA_by_replicate.png")
plot_time(P, pc1, pc2, "PCA_by_time.png")

plot_system(U, "UMAP1", "UMAP2", "UMAP_by_system.png")
plot_replicate(U, "UMAP1", "UMAP2", "UMAP_by_replicate.png")
plot_time(U, "UMAP1", "UMAP2", "UMAP_by_time.png")

###################
# Replicate means
###################

rep_rows = []
rep_feature_means = []

for system in SYSTEMS:
    for rep in REPLICATES:
        idx = np.where((meta["system"] == system) & (meta["replicate"] == rep))[0]
        if not len(idx):
            raise ValueError(f"Missing frames for {system} clone {rep}")
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"Noncontiguous frames for {system} clone {rep}")
        feature_mean = X[idx[0]:idx[-1] + 1].mean(axis=0)
        rep_feature_means.append(feature_mean)

        rep_rows.append({
            "system": system,
            "replicate": rep,
            "overall_hbond_mean": feature_mean.mean(),
            "PC1_mean": P[idx, 0].mean(),
            "PC2_mean": P[idx, 1].mean(),
            "UMAP1_mean": U[idx, 0].mean(),
            "UMAP2_mean": U[idx, 1].mean()
        })

rep = pd.DataFrame(rep_rows)
rep_feature_means = np.asarray(rep_feature_means)
rep.to_csv(OUT / "SETD8_replicate_embedding_means.csv", index=False)
np.save(OUT / "SETD8_replicate_feature_means.npy", rep_feature_means)

print("\nReplicate means:")
print(rep.to_string(index=False))

########################
# Replicate-level ANOVA
########################

def run_anova(variable):
    wt = rep.loc[rep["system"] == "WT", variable].to_numpy()
    mut = rep.loc[rep["system"] == "A296T", variable].to_numpy()

    F, p = f_oneway(wt, mut)
    residuals = np.r_[wt - wt.mean(), mut - mut.mean()]
    W, shapiro_p = shapiro(residuals)
    L, levene_p = levene(wt, mut)
    df_error = len(wt) + len(mut) - 2

    return {
        "variable": variable,
        "WT_mean": wt.mean(),
        "A296T_mean": mut.mean(),
        "difference": wt.mean() - mut.mean(),
        "F": F,
        "p": p,
        "eta_squared": F / (F + df_error),
        "shapiro_W": W,
        "shapiro_p": shapiro_p,
        "levene_F": L,
        "levene_p": levene_p
    }

anova = pd.DataFrame([
    run_anova("overall_hbond_mean"),
    run_anova("PC1_mean")
])

anova.to_csv(OUT / "SETD8_replicate_ANOVA.csv", index=False)

print("\nReplicate-level ANOVA:")
print(anova.to_string(index=False))

#####################
# PCA weight lookup
#####################

row_ids = A0[0, 1:, 0]
col_ids = A0[0, 0, 1:]
assert np.array_equal(row_ids, col_ids), "Residue row/column labels do not match"
rows, cols = np.triu_indices(len(row_ids), k=1)
feature_lookup = np.column_stack((row_ids[rows], col_ids[cols]))

assert feature_lookup.shape[0] == weights.shape[1]

for pc in range(2):
    df = pd.DataFrame({
        "residue_1": feature_lookup[:, 0].astype(int),
        "residue_2": feature_lookup[:, 1].astype(int),
        "weight": weights[pc],
        "magnitude": np.abs(weights[pc])
    })

    df["interaction"] = df["residue_1"].astype(str) + "-" + df["residue_2"].astype(str)
    df = df.sort_values("magnitude", ascending=False)

    df.to_csv(OUT / f"SETD8_PC{pc+1}_ranked_weights.csv", index=False)
    df.head(20).to_csv(OUT / f"SETD8_PC{pc+1}_top20_weights.csv", index=False)

    print(f"\nTop PC{pc+1} weights:")
    print(df.head(20).to_string(index=False))

# NFIX-specific paper interaction maps intentionally removed.
# SETD8 residue numbering must be audited before interpreting these loadings.

print("\nDone.")
print("Saved to:", OUT.resolve())
