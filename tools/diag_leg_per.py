"""Diagnostic for the leg sugar -> PER arc (Tastekin et al. 2026, Fig. 9B: LgLG4 -> AN01B004 -> Bract / Sink&Synch
-> Roundup -> MN9). Lists AN01B004's strongest brain paths to MN9 (2-3 hops, FlyWire v783) and writes a Godot probe
file for every neuron on them. usage: diag_leg_per.py prep | cmp <probe csv> [<probe csv> ...]"""
import sys, json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
AN01B004_FW = [720575940612303271, 720575940618066369, 720575940627697664, 720575940629023870,
               720575940634126711, 720575940633302995]       # matched to BANC AN01B004 (data/vnc/vnc_bridge.csv)
MN9 = [720575940660219265, 720575940618238523]
comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0); ids = comp.index.values
idx = pd.Series(np.arange(len(ids)), index=ids)
a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                usecols=["root_id", "cell_type", "side", "top_nt"]).drop_duplicates("root_id").set_index("root_id").reindex(ids)
name = lambda i: f"{a.cell_type.values[i]}_{str(a.side.values[i])[:1]}_{str(a.top_nt.values[i])[:4]}_{i}"
if sys.argv[1] == "prep":
    con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    src = set(idx.reindex(AN01B004_FW).dropna().astype(int)); dst = set(idx.reindex(MN9).astype(int))
    out1 = con[con.Presynaptic_Index.isin(src)].groupby("Postsynaptic_Index").Connectivity.sum()
    in1 = con[con.Postsynaptic_Index.isin(dst)].groupby("Presynaptic_Index").Connectivity.sum()
    hop = con[con.Presynaptic_Index.isin(out1[out1 >= 10].index) & con.Postsynaptic_Index.isin(in1[in1 >= 10].index)]
    paths = [(int(r.Presynaptic_Index), int(r.Postsynaptic_Index), min(out1[r.Presynaptic_Index], r.Connectivity, in1[r.Postsynaptic_Index]))
             for r in hop.itertuples() if r.Connectivity >= 5]
    paths.sort(key=lambda p: -p[2])
    print("AN01B004 -> X -> Y -> MN9, bottleneck synapse counts:")
    for x, y, w in paths[:20]:
        print(f"  {name(x):35s} -> {name(y):35s} {w}")
    direct = [(i, out1[i], in1[i]) for i in out1.index.intersection(in1.index)]
    for i, o, n in sorted(direct, key=lambda t: -min(t[1], t[2]))[:10]:
        print(f"  2-hop via {name(i):35s} AN->X {o}  X->MN9 {n}")
    keep = set(src) | set(dst) | {p[0] for p in paths[:40]} | {p[1] for p in paths[:40]} | {d[0] for d in direct}
    (ROOT / "recordings/leg_per_probe.json").write_text(json.dumps({name(i): [int(i)] for i in sorted(keep)}))
    print(len(keep), "neurons in recordings/leg_per_probe.json")
else:
    ds = []
    for p in sys.argv[2:]:
        d = pd.read_csv(p); ds.append(d[d.t_ms > 1000].drop(columns=["t_ms", "threat"]).mean().rename(pathlib.Path(p).stem))
    print(pd.concat(ds, axis=1).round(1).to_string())
