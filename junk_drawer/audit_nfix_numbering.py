"""Map MDSA-tools NFIX PCA features onto real NFIX residue numbers.

Put this file in MDSA-tools/Manuscript_JCIM, then run:
    python nfix_numbering_simple.py

Reads existing topologies, six cached H-bond arrays, and PCA weights.
Does NOT rebuild arrays or rerun PCA. Saves three CSVs and a readable report.
"""

from pathlib import Path
import csv

import mdtraj as md
import numpy as np

# ------------------------- 1. File locations -------------------------
# The NFIX source-data directory is next to the MDSA-tools repository.
HERE = Path(__file__).resolve().parent              # .../MDSA-tools/Manuscript_JCIM
BASE = HERE.parent.parent                           # .../MDSA_tools_revisions
DATA = BASE / "Trajectories_and_topologies/NFIX_mutants"
ARRAYS = HERE / "Outputs/NFIX_mutants/hbond_arrays"
RESULTS = HERE / "Outputs/NFIX_WT_G147E"
WEIGHTS = RESULTS / "NFIX_WT_G147E_PCA_weights.npy"
OUT = RESULTS / "numbering_audit"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- 2. Biological reference ------------------------
# Canonical human NFIX amino-acid sequence: UniProt Q14938 (502 aa).
# This is the reference used to convert local matrix positions to
# biological positions. It is NOT taken from the simulation topology.
CANONICAL = """MYSPYCLTQD EFHPFIEALL PHVRAFSYTW FNLQARKRKY FKKHEKRMSK DEERAVKDEL
LGEKPEIKQK WASRLLAKLR KDIRPEFRED FVLTITGKKP PCCVLSNPDQ KGKIRRIDCL
RQADKVWRLD LVMVILFKGI PLESTDGERL YKSPQCSNPG LCVQPHHIGV TIKELDLYLA
YFVHTPESGQ SDSSNQQGDA DIKPLPNGHL SFQDCFVTSG VWNVTELVRV SQTPVATASG
PNFSLADLES PSYYNINQVT LGRRSITSPP STSTTKRPKS IDDSEMESPV DDVFYPGTGR
SPAAGSSQSS GWPNDVDAGP ASLKKSGKLD FCSALSSQGS SPRMAFTHHP LPVLAGVRPG
SPRATASALH FPSTSIIQQS SPYFTHPTIR YHHHHGQDSL KEFVQFVCSD GSGQATGQPN
GSGQGKVPGS FLLPPPPPVA RPVPLPMPDS KSTSTAPDGA ALTPPSPSFA TTGASSANRF
VSIGPRDGNF LNIPQQSQSW FL"""
CANONICAL = "".join(CANONICAL.split())
assert len(CANONICAL) == 502 and CANONICAL[146] == "G"

AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCEQGHILKMFPSTWYV",
))
AA3.update({"HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H",
            "HSP": "H", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E",
            "LYN": "K", "MSE": "M"})


def align_to_nfix(topology):
    """Find where this simulated protein fragment starts in canonical NFIX.

    Here the fragment is contiguous (no internal insertions/deletions).
    Slide its sequence along the full 502-aa reference and count matches.
    This tolerates the G147E substitution and small additional mismatches.
    """
    protein = [r for r in topology.residues if r.name.upper() in AA3]
    fragment = "".join(AA3[r.name.upper()] for r in protein)

    candidates = []
    for start in range(len(CANONICAL) - len(fragment) + 1):
        reference_part = CANONICAL[start:start + len(fragment)]
        matches = sum(a == b for a, b in zip(fragment, reference_part))
        candidates.append((matches, start))

    best = max(score for score, start in candidates)
    starts = [start for score, start in candidates if score == best]
    identity = best / len(fragment)
    if len(starts) != 1 or identity < 0.95:
        raise ValueError("Canonical sequence match is unclear; don't guess numbering.")

    start = starts[0]  # zero-based starting position in full reference
    # Index in MDTraj topology -> 1-based biological position in UniProt.
    mapping = {res.index: start + k + 1 for k, res in enumerate(protein)}
    return mapping, len(protein), identity, start + 1


def read_system(system):
    """Check that array labels correspond to the actual topology rows."""
    top_file = DATA / system / "trajectories/ref.gro"
    top = md.load(str(top_file)).topology
    mapping, protein_count, identity, first_position = align_to_nfix(top)

    paths = sorted(ARRAYS.glob(f"{system}_replica*.npy"))
    if len(paths) != 3:
        raise ValueError(f"Expected 3 {system} arrays; found {len(paths)} in {ARRAYS}")

    header = None
    for path in paths:
        arr = np.load(path, mmap_mode="r")  # no need to load a huge array into RAM
        n = top.n_residues
        if arr.ndim != 3 or arr.shape[1:] != (n + 1, n + 1):
            raise ValueError(f"Unexpected matrix shape in {path.name}: {arr.shape}")
        labels = arr[0, 0, 1:]
        expected = np.arange(1, n + 1)
        if not np.array_equal(labels, expected):
            raise ValueError(f"Array labels are not 1..{n}: {path.name}")
        if not np.array_equal(labels, [r.resSeq for r in top.residues]):
            raise ValueError(f"Array labels differ from topology resSeq: {path.name}")
        for frame in (0, len(arr) - 1):
            if not np.array_equal(arr[frame, 0, 1:], labels):
                raise ValueError(f"Header changed in {path.name}, frame {frame}")
            if not np.array_equal(arr[frame, 1:, 0], labels):
                raise ValueError(f"Row labels changed in {path.name}, frame {frame}")
        header = labels.astype(int)

    return dict(top=top, mapping=mapping, protein_count=protein_count,
                identity=identity, first=first_position, paths=paths, header=header)


def amino_acid(system, index):
    """Biological residue name, including mutation (e.g. E147, not G147)."""
    info = systems[system]
    biological_position = info["mapping"].get(int(index))
    if biological_position is None:
        return "nonprotein"
    residue = info["top"].residue(int(index))
    return f"{AA3[residue.name.upper()]}{biological_position}"


def write_csv(filename, rows):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------- 3. Run checks ----------------------------
systems = {name: read_system(name) for name in ("WT", "G147E")}
if not np.array_equal(systems["WT"]["header"], systems["G147E"]["header"]):
    raise ValueError("WT and G147E feature labels differ.")

# Local index 0 maps to matrix row/column 1; that first row/column is labels.
n = systems["WT"]["top"].n_residues
i_indices, j_indices = np.triu_indices(n, k=1)
weights = np.load(WEIGHTS)
assert weights.ndim == 2 and weights.shape[1] == len(i_indices)
order = np.argsort(-np.abs(weights[0]))  # PC1 ranking by absolute magnitude
ranks = np.empty(len(order), dtype=int)
ranks[order] = np.arange(1, len(order) + 1)
feature_columns = {(int(i), int(j)): p for p, (i, j) in
                   enumerate(zip(i_indices, j_indices))}

# Make biological-position -> topology-index lookup.
reverse = {sys: {position: index for index, position in data["mapping"].items()}
           for sys, data in systems.items()}

report = []
def say(message=""):
    print(message)
    report.append(message)

say("NFIX RESIDUE NUMBERING — SIMPLE AUDIT")
say("=" * 50)
for sys, data in systems.items():
    say(f"{sys}: {data['protein_count']} simulated protein residues; "
        f"first maps to canonical NFIX position {data['first']}; "
        f"sequence agreement {data['identity']:.1%}; "
        f"array/topology labels PASS")

say("\nEXAMPLE: HOW THE NUMBERING WORKS")
for local_label in (104, 116, 135):
    index = local_label - 1     # first row/column is labels, local numbering starts at 1
    say(f"Matrix label {local_label:3d} -> "
        f"WT {amino_acid('WT', index):5s} / "
        f"mutant {amino_acid('G147E', index):5s}")

# Extra biological sanity check: actual topology residue at mutation site.
assert amino_acid("WT", reverse["WT"][147]) == "G147"
assert amino_acid("G147E", reverse["G147E"][147]) == "E147"
say("Mutation identity: WT G147 / mutant E147 confirmed from topology")

# Save the full local -> canonical mapping so every residue can be checked.
residue_rows = []
for sys, data in systems.items():
    for residue in data["top"].residues:
        canonical_position = data["mapping"].get(residue.index)
        residue_rows.append({
            "system": sys,
            "matrix_label": residue.index + 1,
            "topology_resSeq": residue.resSeq,
            "topology_amino_acid": AA3.get(residue.name.upper(), residue.name),
            "canonical_NFIX_position": canonical_position or "",
            "canonical_reference_amino_acid": CANONICAL[canonical_position - 1] if canonical_position else "",
        })
write_csv("residue_numbering.csv", residue_rows)

# ---------------------- 4. Rank PCA features ------------------------
feature_rows = []
for p in range(len(order)):
    i, j = int(i_indices[p]), int(j_indices[p])
    feature_rows.append({
        "PC1_rank": int(ranks[p]),
        "matrix_pair": f"{i + 1}-{j + 1}",
        "WT_biological_pair": f"{amino_acid('WT', i)}-{amino_acid('WT', j)}",
        "G147E_biological_pair": f"{amino_acid('G147E', i)}-{amino_acid('G147E', j)}",
        "PC1_weight": float(weights[0, p]),
        "PC2_weight": float(weights[1, p]),
    })
write_csv("all_mapped_pca_features.csv", sorted(feature_rows, key=lambda x: x["PC1_rank"]))
say("\nTOP 10 PC1 INTERACTIONS (ranked by |weight|)")
for row in sorted(feature_rows, key=lambda x: x["PC1_rank"])[:10]:
    say(f"#{row['PC1_rank']:2d} matrix {row['matrix_pair']:>7s} -> "
        f"WT {row['WT_biological_pair']:<12s} / "
        f"mutant {row['G147E_biological_pair']:<12s} "
        f"(PC1 weight {row['PC1_weight']:+.3f})")

# ------------------ 5. Check published contacts ---------------------
contact_rows = []
say("\nCONTACTS DISCUSSED IN THE NFIX PAPER")
for first, second in ((116, 147), (128, 147)):
    # The WT and mutant arrays must refer to the same feature column.
    pair_wt = sorted((reverse["WT"][first], reverse["WT"][second]))
    pair_mut = sorted((reverse["G147E"][first], reverse["G147E"][second]))
    if pair_wt != pair_mut:
        raise ValueError("Canonical position mapping differs between WT and mutant")
    i, j = pair_wt
    col = feature_columns[(i, j)]
    say(f"\nCanonical positions {first}–{second}: "
        f"PC1 rank #{ranks[col]} / {len(order)}")
    for sys, data in systems.items():
        values = []
        for rep, path in enumerate(data["paths"], start=1):
            arr = np.load(path, mmap_mode="r")
            # +1 because matrix row 0 and column 0 hold residue labels.
            mean = float(np.mean(arr[:, i + 1, j + 1], dtype=np.float64))
            values.append(mean)
            contact_rows.append({
                "canonical_positions": f"{first}-{second}",
                "system": sys,
                "biological_pair": f"{amino_acid(sys, i)}-{amino_acid(sys, j)}",
                "replicate": rep,
                "mean_hbonds_per_frame": mean,
                "PC1_rank": int(ranks[col]),
            })
        say(f"  {sys:5s} {amino_acid(sys, i)}–{amino_acid(sys, j)}: "
            f"mean {np.mean(values):.4f} H-bonds/frame "
            f"(replicates: {', '.join(f'{v:.4f}' for v in values)})")

write_csv("published_contacts_by_replicate.csv", contact_rows)

# --------------------------- 6. Save -------------------------------
say("\nIMPORTANT INTERPRETATION NOTES")
say("- Topology identifies simulated residues; sequence alignment provides canonical numbers.")
say("- Positive/negative PCA loading signs do NOT directly prove bonds formed or broke.")
say("- A residue-level H-bond count does NOT establish a salt bridge.")
say("- Label checks PASS, but do not independently verify the original array-generating code.")
say("- These are exploratory comparisons based on 3 independent runs per system.")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "READ_ME_FIRST.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
say(f"\nSaved report and CSVs in: {OUT}")
