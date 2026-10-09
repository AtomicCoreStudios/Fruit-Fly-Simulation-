"""Axis and direction of the neck proprioceptors from their wiring onto the real neck motor neurons.
Criterion as for the leg FeCO claw/hook (tools/build_leg_map.py, Lee et al. 2025): signed 1-hop + 2-hop drive
(row-normalised BANC weights, via excited interneurons) of each afferent onto each axis' + pool (left neck motor
neurons, turning/rolling the head to the left) minus its - pool (data/neck_motor_map.json).
Axis = the axis with the larger |score|. Direction (prosternal organ hair plates, SNpp19): negative feedback, as the
prosternal organ's role in head-position control (Preuss & Hengstenberg 1992): an afferent that drives the + pool
fires when the head is turned/rolled to the - side ("neg"), and vice versa. Score 0: 'none' (fires at either limit).
Neck chordotonal afferents are speed sensors here (axis only). basis: measured wiring + published function.
(2026-10-09: a full-network settle test was tried first and rejected: every afferent pushed the head to the right,
because the left DProN1 has only 231 annotated input synapses vs 1,438 on the right, a reconstruction asymmetry.)
usage: .venv/Scripts/python tools/neck_sensor_tuning.py -> data/neck_sensor_tuning.json"""
import json, pathlib
from collections import Counter
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = pathlib.Path(__file__).resolve().parents[1]
nm = json.load(open(ROOT / "data/neck_motor_map.json"))
e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet")
ids = np.unique(np.concatenate([e.Presynaptic_Index, e.Postsynaptic_Index])); pos = {x: i for i, x in enumerate(ids)}
W = sp.csr_matrix((e["Excitatory x Connectivity"].astype(float).values,
                   ([pos[x] for x in e.Postsynaptic_Index], [pos[x] for x in e.Presynaptic_Index])), shape=(len(ids), len(ids)))
Wn = W.multiply(1.0 / np.maximum(np.abs(W).sum(axis=1), 1.0)).tocsr()
AXES = ["yaw", "roll"]
res = {}
for kind in ("prosternal", "neck_chordotonal"):
    for x in nm["sensory"][kind]:
        v0 = np.zeros(len(ids)); v0[pos[x]] = 1.0
        h1 = Wn @ v0; h = h1 + Wn @ np.maximum(h1, 0.0)
        sc = [h[[pos[m] for m in nm["motor"][a]["pos"]]].sum() - h[[pos[m] for m in nm["motor"][a]["neg"]]].sum() for a in AXES]
        ai = int(np.argmax(np.abs(sc)))
        lim = "neg" if sc[ai] > 0 else ("pos" if sc[ai] < 0 else "none")
        res[str(x)] = {"kind": kind, "axis": AXES[ai], "limit": lim if kind == "prosternal" else "none",
                       "score": [round(float(s), 5) for s in sc]}
json.dump(res, open(ROOT / "data/neck_sensor_tuning.json", "w"), indent=0)
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv").set_index("model_index")
print(Counter((r["kind"], v.side[int(k)], r["axis"], r["limit"], r["score"] != [0.0, 0.0]) for k, r in res.items()))
