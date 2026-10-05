"""Cable length of every BANC neuron from the pcg-skel SWC skeletons (Bates et al. 2025, Harvard Dataverse
doi:10.7910/DVN/8TFGGB, neuron_skeletons.zip, v626, CC-BY-4.0) -> data/vnc/banc_cable_length.csv
(root_id, cable_um). Skeleton radii are constant (1.0), so surface area is predicted from cable length with a
fit to the measured BANC surface areas of Pugliese et al. (data/raw/pugliese)."""
import io, zipfile, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
z = zipfile.ZipFile(ROOT / "data/raw/banc/neuron_skeletons.zip")
rows = []
for name in z.namelist():
    if not name.endswith(".swc"):
        continue
    a = np.loadtxt(io.BytesIO(z.read(name)), usecols=(0, 2, 3, 4, 6), ndmin=2)
    if len(a) < 2:
        continue
    idx = {int(k): i for i, k in enumerate(a[:, 0])}
    par = np.array([idx.get(int(p), -1) for p in a[:, 4]])
    m = par >= 0
    L = np.linalg.norm(a[m, 1:4] - a[par[m], 1:4], axis=1).sum()
    rows.append((int(pathlib.Path(name).stem), L))
d = pd.DataFrame(rows, columns=["root_id", "cable_um"])
d.to_csv(ROOT / "data/vnc/banc_cable_length.csv", index=False)
print(len(d), "skeletons; cable length median", round(d.cable_um.median()), "um")
pt = pd.read_csv(ROOT / "data/raw/pugliese/wTable_20260217_fullData_consistentColumns.csv")
j = pt.merge(d, left_on="pt_root_id", right_on="root_id")
j = j[(j.surf_area_um2 > 0) & (j.cable_um > 0)]
k, c = np.polyfit(np.log(j.cable_um), np.log(j.surf_area_um2), 1)
r = np.corrcoef(np.log(j.cable_um), np.log(j.surf_area_um2))[0, 1]
print(f"fit on {len(j)} neurons: log(area um2) = {k:.3f} log(cable um) + {c:.3f}, r = {r:.3f}")
