"""Frame-level SETD8 WT vs A296T CONTACT PCA (post-750-ns frames).

Contacts: minimum inter-residue heavy-atom distance < 0.4 nm;
exclude residue pairs separated by fewer than 3 sequence positions.

IMPORTANT: The earlier contact-vs-H-bond script cached only *per-clone mean
occupancies*. It did not save per-frame contacts, so those means cannot be
used for PCA. This script computes and caches the per-frame binary contacts
once. Subsequent runs reuse those caches.

Put this script in Manuscript_JCIM and run: python -u SETD8_contact_PCA.py
"""
from pathlib import Path
import os

import colorcet as cc
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import tables
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------- Settings --------------------
DATA = Path(r"C:/Users/Luis/Projects/MDSA_tools_revisions/Trajectories_and_topologies/SETD8")
BASE = Path(__file__).resolve().parent / "Outputs"
OUT = BASE / "SETD8_WT_A296T_contact_PCA_post750ns"
CACHE = OUT / "per_clone_frame_contacts"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SYSTEMS = {"WT": 0, "A296T": 9}
CLONES = range(30)
CUTOFF_PS = 750_000             # absolute simulation timestamp: 750 ns
CONTACT_NM = 0.4                # contact if closest-heavy distance < 4 Å
CHUNK = 50                      # frames per distance calculation

FEATURES = OUT / "SETD8_contact_feature_matrix.npy"
SCORES = OUT / "SETD8_contact_PCA.npy"
WEIGHTS = OUT / "SETD8_contact_PCA_weights.npy"
VARIANCE = OUT / "SETD8_contact_PCA_variance.npy"

# -------------------- Residue pairs / topology --------------------
first_h5 = DATA / "WT" / "run0-clone0.h5"
first_top = md.load_frame(str(first_h5), index=0).topology
n_res = first_top.n_residues
pairs = np.array([(i, j) for i in range(n_res)
                  for j in range(i + 3, n_res)], dtype=np.int32)
assert n_res == 162 and len(pairs) == 12720, "Unexpected residue/contact pair count"
print(f"Residues: {n_res}; eligible contact pairs: {len(pairs)}", flush=True)

# -------------------- Inspect all clones / count retained frames --------------------
# Reading HDF5 timestamps does not load trajectory coordinates.
rows = []
clones = []
signatures = {}
for system, run in SYSTEMS.items():
    for clone in CLONES:
        h5 = DATA / system / f"run{run}-clone{clone}.h5"
        if not h5.is_file():
            raise FileNotFoundError(h5)
        top = md.load_frame(str(h5), index=0).topology
        signature = tuple((r.resSeq, r.name) for r in top.residues)
        if len(signature) != n_res:
            raise ValueError(f"Unexpected residue count: {h5}")
        if system not in signatures:
            signatures[system] = signature
        elif signature != signatures[system]:
            raise ValueError(f"Different topology within {system}: {h5}")

        with tables.open_file(str(h5), mode="r") as f:
            time_ps = np.asarray(f.root.time[:])
        if not np.all(np.diff(time_ps) >= 0):
            raise ValueError(f"Unsorted timestamps: {h5}")
        keep = np.flatnonzero(time_ps >= CUTOFF_PS)
        if not len(keep):
            raise ValueError(f"No post-750-ns frames in {h5}")
        clones.append((system, clone, h5, len(time_ps), len(keep)))
        rows.extend({"system": system, "replicate": clone,
                     "frame": int(k), "time_ns": float(time_ps[k] / 1000)}
                    for k in keep)

wt, mut = signatures["WT"], signatures["A296T"]
changes = [(idx, a, b) for idx, (a, b) in enumerate(zip(wt, mut)) if a != b]
if any(a[0] != b[0] for a, b in zip(wt, mut)):
    raise ValueError("WT/A296T topology numbering differs")
if len(changes) != 1 or (changes[0][1][1], changes[0][2][1]) != ("ALA", "THR"):
    raise ValueError(f"Unexpected WT/A296T residue changes: {changes}")
print("Mutation position (0-based, WT, A296T):", changes, flush=True)

resseq = np.array([res[0] for res in wt])
meta = pd.DataFrame(rows)
meta.to_csv(OUT / "SETD8_contact_frame_metadata.csv", index=False)
pd.DataFrame({"index_0based": range(n_res), "mdtraj_resSeq": resseq,
              "WT_residue": [r[1] for r in wt],
              "A296T_residue": [r[1] for r in mut]}).to_csv(
                  OUT / "SETD8_contact_residue_numbering.csv", index=False)
print(f"Retained frames: {len(meta)}; group counts: {meta.system.value_counts().to_dict()}", flush=True)

# Check that we're comparing exactly the same frames as the existing H-bond PCA.
hb_meta = BASE / "SETD8_WT_A296T_post750ns" / "SETD8_WT_A296T_frame_results.csv"
if hb_meta.exists():
    hb = pd.read_csv(hb_meta, usecols=["system", "replicate", "frame", "time_ns"])
    same_ids = hb[["system", "replicate", "frame"]].equals(
        meta[["system", "replicate", "frame"]])
    same_times = len(hb) == len(meta) and np.allclose(hb.time_ns, meta.time_ns)
    if not (same_ids and same_times):
        raise ValueError("Contact frames differ from post-750-ns H-bond analysis")
    print("Frame alignment with existing H-bond PCA: PASS", flush=True)

# -------------------- Compute / reuse per-FRAME contact caches --------------------
for system, clone, h5, n_total, n_keep in clones:
    cache = CACHE / f"{system}_clone{clone}.npy"
    if cache.exists():
        check = np.load(cache, mmap_mode="r")
        if check.shape != (n_keep, len(pairs)) or check.dtype != np.uint8:
            raise ValueError(f"Invalid per-frame contact cache: {cache}")
        print(f"{system} clone {clone}: reused {n_keep}-frame contact cache", flush=True)
        continue

    # Write as uint8: only ~6–8 MB per clone, with no distance matrices retained.
    temp = CACHE / f"{system}_clone{clone}.partial.npy"
    contact_frames = np.lib.format.open_memmap(
        temp, mode="w+", dtype=np.uint8, shape=(n_keep, len(pairs)))
    offset = 0
    written = 0
    for chunk in md.iterload(str(h5), chunk=CHUNK):
        kept = np.flatnonzero(chunk.time >= CUTOFF_PS)
        if len(kept):
            distances, returned_pairs = md.compute_contacts(
                chunk[kept], contacts=pairs, scheme="closest-heavy", periodic=True)
            if not np.array_equal(returned_pairs, pairs):
                raise ValueError("Unexpected contact-pair order")
            contact_frames[written:written + len(kept)] = distances < CONTACT_NM
            written += len(kept)
        offset += chunk.n_frames

    if offset != n_total or written != n_keep:
        raise ValueError(f"Frame count mismatch for {h5}: {written}/{n_keep}")
    contact_frames.flush()
    del contact_frames
    os.replace(temp, cache)
    print(f"{system} clone {clone}: generated {written}/{n_total} frame contacts", flush=True)

# -------------------- Build full feature matrix (original clone ordering) --------------------
n_frames, n_features = len(meta), len(pairs)
if FEATURES.exists():
    X = np.load(FEATURES, mmap_mode="r")
    if X.shape != (n_frames, n_features) or X.dtype != np.float32:
        raise ValueError(f"Incompatible combined feature matrix: {FEATURES}")
    print("Reused feature matrix:", X.shape, flush=True)
else:
    print(f"Building feature matrix: {n_frames:,} x {n_features:,} "
          f"(~{n_frames*n_features*4/1024**3:.2f} GiB float32)", flush=True)
    temp = OUT / "SETD8_contact_feature_matrix.partial.npy"
    X = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                  shape=(n_frames, n_features))
    start = 0
    for system, clone, h5, n_total, n_keep in clones:
        c = np.load(CACHE / f"{system}_clone{clone}.npy", mmap_mode="r")
        X[start:start + n_keep] = c
        start += n_keep
    assert start == n_frames
    X.flush()
    del X
    os.replace(temp, FEATURES)
    X = np.load(FEATURES, mmap_mode="r")

# -------------------- PCA: frames are observations, contacts are features --------------------
if SCORES.exists() and WEIGHTS.exists() and VARIANCE.exists():
    P = np.load(SCORES)
    weights = np.load(WEIGHTS)
    var = np.load(VARIANCE)
else:
    # Randomized PCA centers columns like standard PCA but is practical at this scale.
    print("Fitting contact PCA (randomized SVD, 2 components)...", flush=True)
    model = PCA(n_components=2, svd_solver="randomized", random_state=42)
    P = model.fit_transform(X)
    weights = model.components_
    var = model.explained_variance_ratio_
    np.save(SCORES, P)
    np.save(WEIGHTS, weights)
    np.save(VARIANCE, var)

assert P.shape == (n_frames, 2) and weights.shape == (2, n_features)
meta["PC1"] = P[:, 0]
meta["PC2"] = P[:, 1]
meta.to_csv(OUT / "SETD8_contact_frame_results.csv", index=False)

labels = (meta.system == "A296T").to_numpy().astype(int)
sil = silhouette_score(P, labels, sample_size=min(3000, n_frames), random_state=42)
print("Contact PCA variance:", var, "silhouette:", sil, flush=True)

# -------------------- Clone-level summaries --------------------
rep = meta.groupby(["system", "replicate"], sort=False)[["PC1", "PC2"]].mean().reset_index()
rep.to_csv(OUT / "SETD8_contact_replicate_PCA_means.csv", index=False)
wt_pc1 = rep.loc[rep.system == "WT", "PC1"].to_numpy()
mut_pc1 = rep.loc[rep.system == "A296T", "PC1"].to_numpy()
F, p = f_oneway(wt_pc1, mut_pc1)
print(f"Clone-level PC1 ANOVA: F={F:.5g}, p={p:.5g}", flush=True)

pd.DataFrame([{"WT_frames": int((labels == 0).sum()),
               "A296T_frames": int((labels == 1).sum()),
               "PCA_PC1_variance": var[0], "PCA_PC2_variance": var[1],
               "PCA_silhouette_sampled_3000": sil,
               "replicate_PC1_F": F, "replicate_PC1_p": p}]).to_csv(
                   OUT / "SETD8_contact_PCA_summary.csv", index=False)

# -------------------- PCA loading tables with REAL topology residue numbers --------------------
for pc in range(2):
    rank = pd.DataFrame({
        "index_1_0based": pairs[:, 0], "index_2_0based": pairs[:, 1],
        "resSeq_1": resseq[pairs[:, 0]], "resSeq_2": resseq[pairs[:, 1]],
        "weight": weights[pc], "magnitude": np.abs(weights[pc])
    }).sort_values("magnitude", ascending=False)
    rank.to_csv(OUT / f"SETD8_contact_PC{pc+1}_ranked_weights.csv", index=False)
    rank.head(20).to_csv(OUT / f"SETD8_contact_PC{pc+1}_top20_weights.csv", index=False)
    print(f"Top PC{pc+1} contacts:\n", rank.head(10).to_string(index=False), flush=True)

# -------------------- Figures --------------------
for system in SYSTEMS:
    idx = meta.system.eq(system).to_numpy()
    plt.scatter(P[idx, 0], P[idx, 1], s=5, alpha=0.35, label=system)
plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "contact_PCA_by_system.png", dpi=300)
plt.close()

plt.figure(figsize=(11, 7))
for color, ((system, clone), group) in zip(
        cc.glasbey[:60], meta.groupby(["system", "replicate"], sort=False)):
    idx = group.index.to_numpy()
    plt.scatter(P[idx, 0], P[idx, 1], s=5, color=color,
                label=f"{system}_clone{clone}", alpha=0.4)
plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.legend(fontsize=6, ncol=3, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(OUT / "contact_PCA_by_replicate.png", dpi=300)
plt.close()

plt.scatter(P[:, 0], P[:, 1], c=meta.time_ns, s=5)
plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.colorbar(label="Simulation time (ns)")
plt.tight_layout()
plt.savefig(OUT / "contact_PCA_by_time.png", dpi=300)
plt.close()

print("\nDone. Saved to:", OUT.resolve(), flush=True)
