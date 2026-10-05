"""Which inputs drive a neuron at rest? prep: write a probe of the top presynaptic partners (by |weight|, current
build) of the given model indices; cmp <probe csv>: mean drive = rate x weight per partner (mV/s units of the LIF).
usage: diag_rest_drive.py prep i,j,...  |  cmp <csv>"""
import sys, json, struct, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
f = open(ROOT / "data/connectome_flywire.bin", "rb"); assert f.read(4) == b"FLYC"
_, n, e, n_pad, n_groups = struct.unpack("<iiiii", f.read(20))
f.seek(24 + 2 * n * 4)
row = np.frombuffer(f.read((n + 1) * 4), np.int32); col = np.frombuffer(f.read(e * 4), np.int32)
w = np.fromfile(ROOT / "data/lif_w_graded.bin", np.int32).astype(np.float64)
FIXED = 1000.0
meta = json.load(open(ROOT / "data/connectome_flywire.json"))
pre = np.repeat(np.arange(n), np.diff(row))
names = {}
try:
    c = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "cell_type", "super_class"]).drop_duplicates("root_id").set_index("root_id").reindex(c.index)
    names = {i: f"{t}" for i, t in enumerate(a.cell_type.fillna(a.super_class).values)}
    v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
    names.update({int(r.model_index): f"vnc:{r.cell_type}" for r in v.itertuples()})
except Exception:
    pass
if sys.argv[1] == "prep":
    tg = [int(x) for x in sys.argv[2].split(",")]
    m = np.isin(col, tg)
    d = pd.DataFrame({"pre": pre[m], "post": col[m], "w": w[m] / FIXED})
    top = d.groupby("pre").w.sum().abs().sort_values(ascending=False).head(60).index
    probe = {f"{names.get(i, '?')}_{i}": [int(i)] for i in top}
    probe.update({f"TARGET_{names.get(i, '?')}_{i}": [i] for i in tg})
    (ROOT / "recordings/rest_drive_probe.json").write_text(json.dumps(probe))
    d.to_csv(ROOT / "recordings/rest_drive_edges.csv", index=False)
    print(len(probe), "probe groups")
else:
    d = pd.read_csv(ROOT / "recordings/rest_drive_edges.csv"); r = pd.read_csv(sys.argv[2]); r = r[r.t_ms > 2000]
    rate = {int(k.rsplit("_", 1)[1]): r[k].mean() for k in r.columns if k.rsplit("_", 1)[-1].isdigit() and not k.startswith(("TARGET", "joint_", "stance_", "foot_h_"))}
    print("targets:", {k: round(r[k].mean(), 1) for k in r.columns if k.startswith("TARGET")})
    d["rate"] = d.pre.map(rate)
    d = d.dropna(); d["drive"] = d.w * d.rate
    s = d.groupby("pre").agg(w=("w", "sum"), rate=("rate", "first"), drive=("drive", "sum"))
    s["name"] = [names.get(i, "?") for i in s.index]
    print(s.sort_values("drive", key=abs, ascending=False).head(20).round(2).to_string())
    print("net drive (mV*Hz):", round(s.drive.sum(), 1), " excitatory:", round(s.drive[s.drive > 0].sum(), 1))
