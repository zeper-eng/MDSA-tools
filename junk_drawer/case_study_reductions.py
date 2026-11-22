from pathlib import Path
import matplotlib.pyplot as plt
import tables
import numpy as np
import pandas as pd
from mdsa_tools.Data_gen_hbond import TrajectoryProcessor
from mdsa_tools.Analysis import systems_analysis
from sklearn.metrics import silhouette_score

DATA = Path(r"C:\Users\Luis\Projects\MDSA_tools_revisions\Trajectories_and_topologies\SETD8")
BASE = Path(__file__).resolve().parent / "Outputs"
HBOND_DIR = BASE / "SETD8" / "hbond_arrays"
OUT = BASE / "SETD8_WT_A296T_post750ns"
OUT.mkdir(parents=True, exist_ok=True)
HBOND_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["WT", "A296T"]
RUNS = {"WT": 0, "A296T": 9}
REPLICATES = range(30)  # clone0 through clone29, inclusive
PREFIX = "SETD8_WT_A296T"
FEATURES = OUT / f"{PREFIX}_feature_matrix.npy"
PCA = OUT / f"{PREFIX}_PCA.npy"
UMAP = OUT / f"{PREFIX}_UMAP.npy"
META = OUT / f"{PREFIX}_frame_metadata.csv"
PCA_WEIGHTS = OUT / f"{PREFIX}_PCA_weights.npy"
PCA_VARIANCE = OUT / f"{PREFIX}_PCA_variance.npy"

#####################
# Load/generate data
#####################

arrays = []
metadata = []
reference_ids = None
CUTOFF_PS = 750_000  # 750 ns of simulation time; MDTraj times are in ps

for system in SYSTEMS:
    for rep in REPLICATES:
        h5 = DATA / system / f"run{RUNS[system]}-clone{rep}.h5"
        outfile = HBOND_DIR / f"{system}_clone{rep}.npy"

        if not h5.is_file():
            raise FileNotFoundError(f"Missing trajectory: {h5}")

        # Reuse the full-trajectory hydrogen-bond matrix; do NOT recompute H-bonds.
        if not outfile.is_file():
            raise FileNotFoundError(f"Missing cached H-bond matrix: {outfile}")
        full_A = np.load(outfile, mmap_mode="r")

        # Read only timestamps from the .h5 (not the coordinate trajectory).
        with tables.open_file(str(h5), mode="r") as h5file:
            time_ps = np.asarray(h5file.root.time[:])
        if len(time_ps) != len(full_A):
            raise ValueError(f"Timestamps/cache frame count mismatch for {system} clone {rep}")
        if not np.all(np.diff(time_ps) >= 0):
            raise ValueError(f"Timestamps out of order: {h5}")

        start_frame = int(np.searchsorted(time_ps, CUTOFF_PS, side="left"))
        if start_frame >= len(full_A):
            raise ValueError(f"No frames after 750 ns: {h5}")
        A = full_A[start_frame:]  # contiguous memory-mapped view; no large copy
        print(f"{system} clone {rep}: kept {len(A)}/{len(full_A)} cached frames "
              f"({time_ps[start_frame]/1000:.1f}–{time_ps[-1]/1000:.1f} ns)",
              flush=True)

        # Compare the residue labels embedded in the adjacency matrices.
        row_ids = np.asarray(A[0, 1:, 0])
        col_ids = np.asarray(A[0, 0, 1:])
        if not np.array_equal(row_ids, col_ids):
            raise ValueError(f"Row/column residue labels disagree: {outfile}")
        if reference_ids is None:
            reference_ids = row_ids.copy()
        elif not np.array_equal(row_ids, reference_ids):
            raise ValueError(f"Incompatible residue ordering: {outfile}")

        arrays.append(A)
        metadata.extend(
            {"system": system, "replicate": rep, "frame": i,
             "time_ns": float(time_ps[i] / 1000)}
            for i in range(start_frame, len(full_A))
        )

shapes = {A.shape[1:] for A in arrays}
assert len(shapes) == 1, f"H-bond matrix dimensions differ: {shapes}"

meta = pd.DataFrame(metadata)
meta.to_csv(META, index=False)

print("\nFrames after 750 ns:", len(meta), flush=True)
print("Systems:", meta["system"].value_counts().to_dict(), flush=True)
print("Replicates:", meta.groupby("system")["replicate"].nunique().to_dict(), flush=True)
print("Matrix shape:", next(iter(shapes)), flush=True)

#################
# MDSA analysis
#################

SA = systems_analysis(arrays)

if FEATURES.exists():
    X = np.load(FEATURES, mmap_mode="r")
else:
    n_features = len(reference_ids) * (len(reference_ids) - 1) // 2
    estimated_gb = len(meta) * n_features * 4 / (1024 ** 3)
    print(f"Expected float32 feature matrix: ~{estimated_gb:.2f} GiB", flush=True)
    if estimated_gb > 4:
        print("WARNING: full PCA/UMAP may require considerably more RAM than this estimate.",
              flush=True)
    X = SA.replicates_to_feature_matrix().astype(np.float32)
    np.save(FEATURES, X)

print("Feature matrix:", X.shape, flush=True)
assert X.shape[0] == len(meta), "Frame metadata does not match feature matrix"

labels = pd.Categorical(meta["system"], categories=SYSTEMS).codes

#######
# PCA
#######

if PCA.exists() and PCA_WEIGHTS.exists() and PCA_VARIANCE.exists():
    P = np.load(PCA)
    weights = np.load(PCA_WEIGHTS)
    var = np.load(PCA_VARIANCE)
else:
    P, weights, var = SA.reduce_systems_representations(method="PCA")
    np.save(PCA, P)
    np.save(PCA_WEIGHTS, weights)
    np.save(PCA_VARIANCE, var)

assert len(P) == len(meta), "PCA rows do not match trajectory frames"
meta["PC1"] = P[:, 0]
meta["PC2"] = P[:, 1]
# silhouette_score otherwise computes O(n_frames^2) pairwise distances
pca_silhouette = silhouette_score(P[:, :2], labels,
                                  sample_size=min(3000, len(labels)), random_state=42)

print("\nPCA variance:", var[:2], flush=True)
print("PCA total PC1+PC2 variance:", var[:2].sum(), flush=True)
print("PCA WT vs A296T silhouette (sampled):", pca_silhouette, flush=True)

plt.figure()
for system in SYSTEMS:
    idx = meta["system"].eq(system).to_numpy()
    plt.scatter(P[idx, 0], P[idx, 1], s=5, label=system)

plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "SETD8_WT_vs_A296T_PCA.png", dpi=300)
plt.close()

########
# UMAP
########

if UMAP.exists():
    U = np.load(UMAP)
else:
    U = SA.reduce_systems_representations(method="UMAP")
    np.save(UMAP, U)

assert len(U) == len(meta), "UMAP rows do not match trajectory frames"
meta["UMAP1"] = U[:, 0]
meta["UMAP2"] = U[:, 1]
umap_silhouette = silhouette_score(U, labels,
                                   sample_size=min(3000, len(labels)), random_state=42)

print("\nUMAP WT vs A296T silhouette (sampled):", umap_silhouette, flush=True)

plt.figure()
for system in SYSTEMS:
    idx = meta["system"].eq(system).to_numpy()
    plt.scatter(U[idx, 0], U[idx, 1], s=5, label=system)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "SETD8_WT_vs_A296T_UMAP.png", dpi=300)
plt.close()

########
# Save
########

meta.to_csv(OUT / f"{PREFIX}_frame_results.csv", index=False)

summary = pd.DataFrame([{
    "WT_frames": (meta["system"] == "WT").sum(),
    "A296T_frames": (meta["system"] == "A296T").sum(),
    "PCA_PC1_variance": var[0],
    "PCA_PC2_variance": var[1],
    "PCA_PC1_PC2_variance": var[:2].sum(),
    "PCA_silhouette_sampled_3000": pca_silhouette,
    "UMAP_silhouette_sampled_3000": umap_silhouette,
}])

summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False)

print("\nDone.", flush=True)
print(summary.to_string(index=False), flush=True)
print("\nSaved to:", OUT.resolve(), flush=True)
