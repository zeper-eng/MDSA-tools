"""Compare published-style SETD8 heavy-atom contacts with cached MDSA H-bonds.

Run from Manuscript_JCIM (or place the script there).
WT = OSF 11707 run0 clones 0..29; A296T = OSF 11713 run9 clones 0..29.
Published contact method: closest heavy-atom distance < 0.4 nm, sequence gap >= 3,
frames at or after absolute simulation time 750 ns. Clones weighted equally by
DEFAULT; pooled-frame results are also exported for closer paper comparison.
This script does not run PCA/UMAP or recreate H-bond matrices.
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Luis/Projects/MDSA_tools_revisions/Trajectories_and_topologies/SETD8")
BASE = Path(__file__).resolve().parent / "Outputs"
HBOND_DIR = BASE / "SETD8" / "hbond_arrays"
OUT = BASE / "SETD8_WT_A296T_contact_comparison"
SYSTEMS = {"WT": 0, "A296T": 9}
N_CLONES = 30
CUTOFF_NS = 750.0
CUTOFF_NM = 0.4
CHUNK = 50


def check_signature(system, signature, reference_signatures):
    """Require stable per-system residue numbering, composition, and order."""
    previous = reference_signatures.get(system)
    if previous is None:
        reference_signatures[system] = signature
    elif previous != signature:
        raise ValueError(f"Residue topology differs between {system} clones")


def process_clone(system, run, clone, pairs, matrix_labels, signatures, overwrite):
    filename = ROOT / system / f"run{run}-clone{clone}.h5"
    cached_hb = HBOND_DIR / f"{system}_clone{clone}.npy"
    outfile = OUT / "per_clone" / f"{system}_clone{clone}_contact_vs_hbond.npz"
    if not filename.is_file() or not cached_hb.is_file():
        raise FileNotFoundError(f"Missing trajectory or existing H-bond cache: {filename} / {cached_hb}")

    if outfile.exists() and not overwrite:
        with np.load(outfile) as x:
            contact_fraction = x["contact_fraction"]
            hbond_fraction = x["hbond_fraction"]
            kept = int(x["n_frames"])
        if len(contact_fraction) != len(pairs) or len(hbond_fraction) != len(pairs):
            raise ValueError(f"Incompatible saved per-clone cache: {outfile}")
        print(f"{system} clone {clone}: reused contact cache ({kept} frames)", flush=True)
        return contact_fraction, hbond_fraction, kept

    A = np.load(cached_hb, mmap_mode="r")
    n_res = len(matrix_labels)
    if A.shape[1:] != (n_res + 1, n_res + 1):
        raise ValueError(f"Unexpected matrix shape {cached_hb}: {A.shape}")
    if not np.array_equal(A[0, 1:, 0], matrix_labels) or not np.array_equal(A[0, 0, 1:], matrix_labels):
        raise ValueError(f"Different H-bond residue labels: {cached_hb}")

    counts_contact = np.zeros(len(pairs), dtype=np.int64)
    counts_hbond = np.zeros(len(pairs), dtype=np.int64)
    offset, kept = 0, 0
    i, j = pairs[:, 0] + 1, pairs[:, 1] + 1  # +1 is the header row/column in H-bond arrays
    for traj in md.iterload(str(filename), chunk=CHUNK):
        residues = list(traj.topology.residues)
        if len(residues) != n_res:
            raise ValueError(f"Topology size differs in {filename}")
        signature = tuple((r.resSeq, r.name) for r in residues)
        check_signature(system, signature, signatures)
        if len(traj.time) != traj.n_frames:
            raise ValueError(f"Missing time metadata in {filename}")

        keep = np.flatnonzero(traj.time >= CUTOFF_NS * 1000.0)
        if keep.size:
            chosen = traj[keep]
            distances, actual_pairs = md.compute_contacts(
                chosen, contacts=pairs, scheme="closest-heavy", periodic=True
            )
            if not np.array_equal(actual_pairs, pairs):
                raise ValueError("MDTraj returned a different residue-pair order")
            counts_contact += (distances < CUTOFF_NM).sum(axis=0)
            hb_chunk = np.asarray(A[offset + keep, :, :])
            counts_hbond += (hb_chunk[:, i, j] > 0).sum(axis=0)
            kept += keep.size
        offset += traj.n_frames

    if offset != A.shape[0]:
        raise ValueError(f"Cached H-bond frame count != .h5 frames: {system} clone {clone}")
    if not kept:
        raise ValueError(f"No frames >= {CUTOFF_NS} ns: {filename}")

    frac_contact = (counts_contact / kept).astype(np.float32)
    frac_hbond = (counts_hbond / kept).astype(np.float32)
    np.savez_compressed(outfile, contact_fraction=frac_contact, hbond_fraction=frac_hbond,
                        n_frames=kept)
    print(f"{system} clone {clone}: analyzed {kept}/{offset} frames", flush=True)
    return frac_contact, frac_hbond, kept


def show_heatmap(values, pairs, n_res, title, path):
    mat = np.full((n_res, n_res), np.nan, dtype=float)
    i, j = pairs.T
    mat[i, j] = values
    mat[j, i] = values
    plt.figure(figsize=(7, 6))
    im = plt.imshow(mat, cmap="coolwarm", vmin=-0.5, vmax=0.5, origin="lower")
    plt.colorbar(im, label="A296T minus WT occupancy")
    plt.xlabel("Residue index within 162-residue construct (0-based)")
    plt.ylabel("Residue index within 162-residue construct (0-based)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="recompute cached per-clone contacts")
    args = ap.parse_args()
    (OUT / "per_clone").mkdir(parents=True, exist_ok=True)

    first = np.load(HBOND_DIR / "WT_clone0.npy", mmap_mode="r")
    labels = np.asarray(first[0, 1:, 0]).astype(int)
    n_res = len(labels)
    if not np.array_equal(first[0, 0, 1:], labels):
        raise ValueError("WT cache residue column labels disagree")
    # Authors exclude i,i+1 and i,i+2: 162 residues => 12,720 pairs.
    pairs = np.array([(i, j) for i in range(n_res) for j in range(i+3, n_res)], dtype=np.int32)
    print(f"Residues: {n_res}, eligible contact pairs: {len(pairs)}", flush=True)
    if n_res != 162 or len(pairs) != 12720:
        raise ValueError("Unexpected SETD8 construct length/contact pair count")

    signatures = {}
    data = {}
    all_rows = []
    for system, run in SYSTEMS.items():
        cf, hf, ns = [], [], []
        for clone in range(N_CLONES):
            c, h, n = process_clone(system, run, clone, pairs, labels, signatures, args.overwrite)
            cf.append(c)
            hf.append(h)
            ns.append(n)
            all_rows.append({"system": system, "clone": clone, "frames_post750": n})
        data[system] = (np.stack(cf), np.stack(hf), np.asarray(ns))
    pd.DataFrame(all_rows).to_csv(OUT / "clone_frame_counts.csv", index=False)

    if signatures.get("WT") and signatures.get("A296T"):
        wt_sig, mut_sig = signatures["WT"], signatures["A296T"]
        mismatches = [(idx, a, b) for idx, (a, b) in enumerate(zip(wt_sig, mut_sig)) if a != b]
        print("WT/A296T topology residue differences (0-based index, WT, mutant):", mismatches)
        if any(a[0] != b[0] for a, b in zip(wt_sig, mut_sig)):
            raise ValueError("WT and mutant residue numbering/order differ")
        if len(mismatches) != 1 or (mismatches[0][1][1], mismatches[0][2][1]) != ("ALA", "THR"):
            raise ValueError("Expected exactly one alanine-to-threonine substitution; inspect topologies")
        positions = pd.DataFrame({
            "index_0based": np.arange(n_res), "matrix_label": labels,
            "mdtraj_resSeq": [r[0] for r in wt_sig],
            "WT_residue_name": [r[1] for r in wt_sig],
            "A296T_residue_name": [r[1] for r in mut_sig],
        })
        positions.to_csv(OUT / "residue_numbering_audit.csv", index=False)
        mutation_idx = mismatches[0][0]
        print("Mutation position:", positions.iloc[mutation_idx].to_dict())
    else:
        print("NOTE: All per-clone caches reused; residue numbering audit requires --overwrite or first-run values.")

    wt_c, wt_h, wt_n = data["WT"]
    mut_c, mut_h, mut_n = data["A296T"]
    # Equal-clone means: each independent simulation has equal weight.
    wc, mc = wt_c.mean(axis=0), mut_c.mean(axis=0)
    wh, mh = wt_h.mean(axis=0), mut_h.mean(axis=0)
    # Pooled-frame alternative, closer to the original paper's averaged-contact-map approach.
    wc_pool = np.average(wt_c, weights=wt_n, axis=0)
    mc_pool = np.average(mut_c, weights=mut_n, axis=0)

    table = pd.DataFrame({
        "index_1_0based": pairs[:, 0], "index_2_0based": pairs[:, 1],
        "matrix_label_1": labels[pairs[:, 0]], "matrix_label_2": labels[pairs[:, 1]],
        "WT_contact_fraction": wc, "A296T_contact_fraction": mc,
        "contact_delta": mc - wc,
        "WT_Hbond_fraction": wh, "A296T_Hbond_fraction": mh,
        "Hbond_delta": mh - wh,
        "contact_delta_pooled_frames": mc_pool - wc_pool,
    })
    if signatures.get("WT"):
        resseq = np.array([r[0] for r in signatures["WT"]])
        table["mdtraj_resSeq_1"] = resseq[pairs[:, 0]]
        table["mdtraj_resSeq_2"] = resseq[pairs[:, 1]]
    table["abs_contact_delta"] = table["contact_delta"].abs()
    table["abs_Hbond_delta"] = table["Hbond_delta"].abs()
    table.to_csv(OUT / "all_residue_pair_occupancies.csv", index=False)
    table.sort_values("abs_contact_delta", ascending=False).head(50).to_csv(
        OUT / "top50_contact_changes.csv", index=False)
    table.sort_values("abs_Hbond_delta", ascending=False).head(50).to_csv(
        OUT / "top50_Hbond_changes.csv", index=False)

    np.savez_compressed(OUT / "per_clone_occupancies.npz", pairs=pairs,
                        WT_contact=wt_c, A296T_contact=mut_c,
                        WT_Hbond=wt_h, A296T_Hbond=mut_h,
                        WT_frames=wt_n, A296T_frames=mut_n)
    show_heatmap(mc-wc, pairs, n_res, "SETD8: heavy-atom contact differences (A296T - WT)",
                 OUT / "contact_difference_map.png")
    show_heatmap(mh-wh, pairs, n_res, "SETD8: hydrogen-bond occupancy differences (A296T - WT)",
                 OUT / "Hbond_difference_map.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(mc-wc, mh-wh, s=4, alpha=0.35)
    plt.axhline(0, linewidth=0.7)
    plt.axvline(0, linewidth=0.7)
    plt.xlabel("Heavy-atom contact occupancy change (A296T - WT)")
    plt.ylabel("H-bond occupancy change (A296T - WT)")
    plt.tight_layout()
    plt.savefig(OUT / "contact_vs_Hbond_changes.png", dpi=250)
    plt.close()

    print("\nLargest CONTACT changes:\n", table.nlargest(10, "abs_contact_delta")[
        ["matrix_label_1", "matrix_label_2", "contact_delta", "Hbond_delta"]
    ].to_string(index=False))
    print("\nLargest HBOND changes:\n", table.nlargest(10, "abs_Hbond_delta")[
        ["matrix_label_1", "matrix_label_2", "contact_delta", "Hbond_delta"]
    ].to_string(index=False))
    print("\nSaved:", OUT.resolve())


if __name__ == "__main__":
    main()
