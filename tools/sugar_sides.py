"""Full labellar sugar (LB3c) and sugar/low-salt (LB3b) GRN sets per annotated side -> MN9 (reference LIF)."""
import sys, pathlib, json
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reference_lif import LIF, MN9, ROOT
g = pd.read_csv(ROOT / "data/taste_neurons.csv")
m = LIF(sys.argv[1] if len(sys.argv) > 1 else "gpu2")
mn9 = m.index.reindex(MN9).values
res = {}
for side in ("left", "right"):
    for typ in (["LB3c"], ["LB3b"], ["LB3c", "LB3b"]):
        ids = g[(g.side == side) & g.cell_type.isin(typ)].root_id
        idx = m.index.reindex(ids).dropna().astype(int).values
        for f in (30, 60, 100):
            r = np.mean([m.run([(idx, f)], seed=k)[mn9] for k in range(2)], 0)
            key = f"{side} {'+'.join(typ)} n={len(idx)} @ {f} Hz"
            res[key] = r.round(1).tolist()
            print(f"{key:34s} -> MN9 R/L {r[0]:6.1f}/{r[1]:6.1f}", flush=True)
(ROOT / "recordings/sugar_sides_mn9.json").write_text(json.dumps(res, indent=1))
