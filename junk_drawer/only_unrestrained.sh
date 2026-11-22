python - <<'PY'
from pathlib import Path
import numpy as np
from mdsa_tools.Convenience import unrestrained_residues

D=Path("../Trajectories_and_topologies").resolve()

for system in ["GCU","CGU"]:
    src_path=D/f"redone_CCU_{system}_Trajectory_array.npy"
    out_path=D/f"redone_unrestrained_CCU_{system}_Trajectory_array.npy"

    src=np.load(src_path,mmap_mode="r")

    labels=src[0,0,1:].astype(int)
    keep=np.r_[0,np.where(np.isin(labels,unrestrained_residues))[0]+1]

    print(f"\n{system}")
    print("Original:",src.shape)
    print("Keeping residues:",len(keep)-1)
    print("Output:",(src.shape[0],len(keep),len(keep)))

    out=np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=src.dtype,
        shape=(src.shape[0],len(keep),len(keep))
    )

    for start in range(0,len(src),50):
        stop=min(start+50,len(src))
        out[start:stop]=src[start:stop][:,keep][:,:,keep]
        print(f"{stop}/{len(src)}",end="\r")

    del out
    print(f"\nSaved: {out_path}")

PY

powershell.exe -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -match "Photoshop" } | Stop-Process -Force'