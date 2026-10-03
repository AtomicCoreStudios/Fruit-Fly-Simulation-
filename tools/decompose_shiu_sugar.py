"""Which GRNs in Shiu et al.'s 21 'sugar' neurons actually drive MN9? Re-typed with flywire_annotations
v3.2.0 (Tastekin et al. 2026), the set contains sugar, sugar/low-salt, high-salt/heavy-metal and
putative-attractive GRNs. Each subgroup is stimulated alone in the reference LIF model
(tools/reference_lif.py), plus the complete corrected right-side sugar set."""
import sys, pathlib, json
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reference_lif import LIF, SUGAR_R, MN9, ROOT

scheme = sys.argv[1] if len(sys.argv) > 1 else "gpu2"
freqs = [float(x) for x in sys.argv[2:]] or [100.0, 200.0]
g = pd.read_csv(ROOT / "data/taste_neurons.csv")
mod = dict(zip(g.root_id, g.modality))
m = LIF(scheme)
sets = {}
for r in SUGAR_R:
    sets.setdefault("shiu:" + str(mod.get(r, "not in v783 GRNs")), []).append(r)
sets["shiu: all 21"] = SUGAR_R
right_sugar = g[(g.organ == "labellum_bristle") & (g.side == "right") & g.modality.isin(["sugar", "sugar/low_salt"])].root_id
sets["corrected: all right sugar + sugar/low_salt"] = list(right_sugar)
right_lb3d = g[(g.organ == "labellum_bristle") & (g.side == "right") & (g.modality == "high_salt/heavy_metal")].root_id
sets["corrected: all right high_salt/heavy_metal"] = list(right_lb3d)
mn9 = m.index.reindex(MN9).values
res = {}
for name, ids in sets.items():
    idx = m.index.reindex(ids).dropna().astype(int).values
    for f in freqs:
        r = np.mean([m.run([(idx, f)], seed=k)[mn9] for k in range(2)], 0)
        res[f"{name} @ {f:g} Hz"] = r.round(1).tolist()
        print(f"{name:48s} n={len(idx):2d}  {f:5.0f} Hz -> MN9 R/L {r[0]:6.1f}/{r[1]:6.1f}", flush=True)
(ROOT / f"recordings/shiu_sugar_decomposition_{scheme}.json").write_text(json.dumps(res, indent=1))
