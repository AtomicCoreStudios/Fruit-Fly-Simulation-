"""Pick neurons to probe individually in the loom test: the columns whose facets look toward the
predator's approach (left eye, azimuth +90 deg, elevation ~+7 deg) vs control columns looking
straight ahead. Writes data/loom_probe.json = {group: [neuron indices]} for main.gd --probe."""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
om = json.loads((ROOT / "data/ommatidia.json").read_text())["facets"]
drive = json.loads((ROOT / "data/eye_drive.json").read_text())
comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
index = pd.Series(np.arange(len(comp)), index=comp.index.astype(np.int64))
dirs = np.array(drive["facet_dir"])            # Godot head-local: +X anterior, +Y dorsal, -Z left
def toward(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.cos(a), np.sin(e), -np.cos(e) * np.sin(a)])
TYPES = ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
probe = {}
for name, az, el in (("loom", 90, 7), ("ctrl", 0, 7)):
    ang = np.degrees(np.arccos(np.clip(dirs @ toward(az, el), -1, 1)))
    sel = [i for i in np.argsort(ang)[:12] if om[i]["neurons"]]
    print(name, "facets", len(sel), "max angle %.1f deg" % ang[sel].max())
    for t in TYPES:
        probe[f"{name}:{t}"] = [int(index[int(r)]) for i in sel for r in om[i]["neurons"].get(t, [])]
(ROOT / "data/loom_probe.json").write_text(json.dumps(probe))
print({k: len(v) for k, v in probe.items()})
