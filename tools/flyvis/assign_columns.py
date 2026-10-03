"""Assign hex columns (p, q) to FlyWire neurons of FlyVis cell types that are missing from the
Codex column map (TmY3, TmY4, Tm5a/b/c/Y, Tm16, Tm28, Tm30, Mi2/3/10-15, Lawf1/2, ...), from
connectivity: the synapse-weighted mean (p, q) of each neuron's partners that DO have a Codex column
(inputs and outputs), rounded to the nearest column. basis: approximate, validated here by
hiding the Codex columns of types that have them and checking how often the method recovers them.

Writes data/extra_columns.csv: root_id, hemisphere, type (FlyVis name), p, q, n_syn, method.
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def assign(ids, con, colmap, ncols):
    """ids: root_ids to place; colmap: root_id -> (hemisphere, p, q) for reference neurons."""
    ref = pd.DataFrame(colmap, columns=["root_id", "hemisphere", "p", "q"]).set_index("root_id")
    e1 = con[con.Postsynaptic_ID.isin(ids) & con.Presynaptic_ID.isin(ref.index)].rename(
        columns={"Postsynaptic_ID": "cell", "Presynaptic_ID": "partner"})
    e2 = con[con.Presynaptic_ID.isin(ids) & con.Postsynaptic_ID.isin(ref.index)].rename(
        columns={"Presynaptic_ID": "cell", "Postsynaptic_ID": "partner"})
    e = pd.concat([e1, e2])[["cell", "partner", "Connectivity"]]
    e = e.join(ref, on="partner")
    # majority hemisphere, then weighted mean p, q within it
    hemi = e.groupby(["cell", "hemisphere"]).Connectivity.sum().reset_index().sort_values("Connectivity").groupby("cell").tail(1)
    e = e.merge(hemi[["cell", "hemisphere"]], on=["cell", "hemisphere"])
    g = e.assign(wp=e.p * e.Connectivity, wq=e.q * e.Connectivity).groupby(["cell", "hemisphere"])
    out = (g[["wp", "wq"]].sum().div(g.Connectivity.sum(), axis=0)).reset_index()
    out["n_syn"] = g.Connectivity.sum().values
    # round to the nearest existing column of that hemisphere
    res = []
    for h, d in out.groupby("hemisphere"):
        C = ncols[h]
        dist = np.abs(d.wp.values[:, None] - C[None, :, 0]) + np.abs(d.wq.values[:, None] - C[None, :, 1])
        k = dist.argmin(1)
        res.append(pd.DataFrame({"root_id": d.cell.values, "hemisphere": h, "p": C[k, 0], "q": C[k, 1],
                                 "n_syn": d.n_syn.values}))
    return pd.concat(res)


def main():
    net = np.load(ROOT / "data/flyvis_net.npz")
    fv_types = list(net["types"])
    cols = pd.read_csv(ROOT / "data/raw/column_assignment.csv.gz")
    ncols = {h: d[["p", "q"]].drop_duplicates().to_numpy() for h, d in cols.groupby("hemisphere")}
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t",
                    usecols=["root_id", "cell_type", "side"], low_memory=False).drop_duplicates("root_id")
    con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    # ---- validation: hide Codex columns of known types and recover them
    test = ["Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Mi1", "Mi4", "Mi9", "T4a", "T5a", "T2", "C3"]
    ok, near, tot = 0, 0, 0
    for t in test:
        tc = cols[cols.type == t]
        ref = cols[cols.type != t][["root_id", "hemisphere", "p", "q"]].values.tolist()
        got = assign(set(tc.root_id), con, ref, ncols).merge(tc[["root_id", "p", "q"]], on="root_id", suffixes=("", "_true"))
        d = np.abs(got.p - got.p_true) + np.abs(got.q - got.q_true)
        ok += (d == 0).sum(); near += (d <= 1).sum(); tot += len(got)
    print(f"validation on {len(test)} Codex-mapped types ({tot} neurons): exact column {ok/tot:.0%}, "
          f"within one column {near/tot:.0%}")
    # ---- assign the FlyVis types that Codex lacks; map FlyWire names to FlyVis names
    missing = [t for t in fv_types if t not in set(cols.type) and not t.startswith(("R", "CT1"))]
    name_map = {}
    fw_types = set(a.cell_type.dropna())
    for t in missing:
        cands = [x for x in fw_types if x == t] or [x for x in fw_types if x.startswith(t) and len(x) <= len(t) + 1]
        if cands:
            for x in cands:
                name_map[x] = t
    ids = a[a.cell_type.isin(name_map)]
    ref = cols[["root_id", "hemisphere", "p", "q"]].values.tolist()
    got = assign(set(ids.root_id), con, ref, ncols)
    got["type"] = got.root_id.map(ids.set_index("root_id").cell_type).map(name_map)
    got["fw_type"] = got.root_id.map(ids.set_index("root_id").cell_type)
    got["method"] = "connectivity"
    got.to_csv(ROOT / "data/extra_columns.csv", index=False)
    summ = got.groupby("type").agg(n=("root_id", "size"), fw=("fw_type", lambda s: ",".join(sorted(set(s)))))
    print("FlyVis types without Codex columns:", missing)
    print("assigned:", summ.to_dict("index"))
    print("not found in FlyWire:", [t for t in missing if t not in set(name_map.values())])


if __name__ == "__main__":
    main()
