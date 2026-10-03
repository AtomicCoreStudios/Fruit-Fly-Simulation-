"""Geometry test independent of FlyVis: drive the FlyWire T4/T5 inputs of LPLC2 with an idealised
radial motion pattern and check radial selectivity of the wiring (Klapoetke et al. 2017: LPLC2
dendritic arms in each lobula-plate layer extend in that layer's preferred direction, so OUTWARD
motion excites). Each T4x/T5x neuron in column (p,q) of the right eye is set to
relu(radial_unit . preferred_direction(x)) inside an annulus around a centre; inward = sign flipped.
Preferred directions (Maisak et al. 2013): a posterior, b anterior, c dorsal, d ventral."""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
om = json.loads((ROOT / "data/ommatidia.json").read_text())["facets"]
dirs = np.array(json.loads((ROOT / "data/eye_drive.json").read_text())["facet_dir"])
az = np.degrees(np.arctan2(-dirs[:, 2], dirs[:, 0])); el = np.degrees(np.arcsin(dirs[:, 1]))
pq_dir = {tuple(f["pq"]): (az[i], el[i]) for i, f in enumerate(om) if f["side"] == "right" and f["pq"]}
cols = pd.read_csv(ROOT / "data/raw/column_assignment.csv.gz")
comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
index = pd.Series(np.arange(len(comp)), index=comp.index.astype(np.int64))
con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_type", "side"]).drop_duplicates("root_id").set_index("root_id").reindex(comp.index)
lplc2 = np.where((a.cell_type == "LPLC2") & (a.side == "right"))[0]
PD = {"a": (-1, 0), "b": (1, 0), "c": (0, 1), "d": (0, -1)}   # (anterior, dorsal) for the right eye
t45 = cols[(cols.hemisphere == "right") & cols.type.str.match(r"T[45][abcd]$")].copy()
t45["idx"] = index.reindex(t45.root_id).values
t45 = t45[t45.apply(lambda r: (r.p, r.q) in pq_dir, axis=1)]
t45["az"] = [pq_dir[(p, q)][0] for p, q in zip(t45.p, t45.q)]; t45["el"] = [pq_dir[(p, q)][1] for p, q in zip(t45.p, t45.q)]
e = con[con.Postsynaptic_Index.isin(lplc2) & con.Presynaptic_Index.isin(t45.idx)]
W = e.pivot_table(index="Postsynaptic_Index", columns="Presynaptic_Index", values="Connectivity", aggfunc="sum", fill_value=0)
t45 = t45.set_index("idx").loc[W.columns]
# each LPLC2's receptive-field centre = synapse-weighted mean position of its T4/T5 inputs
cen = (W.values @ t45[["az", "el"]].values) / W.values.sum(1, keepdims=True)
out_sel = []
for i, (az0, el0) in enumerate(cen):
    ant = (t45.az.values - az0) * np.cos(np.radians(el0)); dor = t45.el.values - el0
    r = np.hypot(ant, dor); ru = np.stack([ant, dor], 1) / np.maximum(r, 1e-6)[:, None]
    pd_ = np.array([PD[t[-1]] for t in t45.type])
    ring = (r > 5) & (r < 40)
    outward = np.maximum((ru * pd_).sum(1), 0) * ring; inward = np.maximum(-(ru * pd_).sum(1), 0) * ring
    out_sel.append((W.values[i] @ outward, W.values[i] @ inward))
out_sel = np.array(out_sel)
ratio = out_sel[:, 0] / np.maximum(out_sel[:, 1], 1e-9)
print(f"{len(lplc2)} right LPLC2 cells; T4/T5 synapses per cell {W.values.sum(1).mean():.0f}")
print(f"outward / inward drive per cell: median {np.median(ratio):.2f}, {np.mean(ratio > 1):.0%} of cells prefer OUTWARD")
