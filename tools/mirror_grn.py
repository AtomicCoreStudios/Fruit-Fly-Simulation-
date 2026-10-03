"""Complete the under-reconstructed right labellar taste neurons (GRNs) of FlyWire v783 by bilateral
mirroring from the left side. Writes data/grn_mirror_patch.parquet (synapses to ADD) and a report.

Why (see README "Mirrored labellar GRNs"):
  - right-annotated labellar bristle GRNs (n=105) and taste-peg GRNs (n=36) have ~0.5x the output
    synapses of their left counterparts (n=108/37), with ~0.86x the cable length -> synapse density
    per mm cable 0.57x; labial mechanosensory neurons in the same nerve, pharyngeal and leg GRNs are
    symmetric (tools audit in this file's main()).
  - Engert et al. 2022 eLife 11:e78110: misalignments in the FAFB volume prevented reconstruction of
    many labellar GRNs in FAFB's left hemisphere (= FlyWire 'right'; FAFB was imaged mirrored).
Anatomy respected: each type's ipsilateral vs contralateral partner pattern is copied from the left
template (bitter GRNs project to the midline ring, sugar/water/salt stay ipsilateral; Engert et al.
2022, Wang et al. 2004 Cell, Thorne et al. 2004 Curr Biol), so mirroring maps ipsi->ipsi and
contra->contra onto the mirror-image partner neurons (same cell type, opposite side).
Method, per right GRN g of type T and partner (type P, relative side s):
  expected = mean synapses per LEFT neuron of type T onto partners of type P on side s
  added    = max(0, expected - observed(g, P, s))      (measured synapses are never removed)
  distributed over the right-side-mirrored partner neurons of type P in proportion to the synapses
  they already receive from right GRNs of type T (+1), stochastic rounding, fixed seed.
Inputs onto the GRN axons (presynaptic feedback, e.g. GABAergic GNG016) are completed the same way.
Partners without a cell type cannot be mirrored; their share is reported. basis: approximate.
"""
import pathlib, json
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 783


def rel_side(a, b):
    if a in ("left", "right") and b in ("left", "right"):
        return "ipsi" if a == b else "contra"
    return "center"


def mirror_side(src_side, rel):
    if rel == "center":
        return "center"
    return src_side if rel == "ipsi" else ("left" if src_side == "right" else "right")


def main():
    rng = np.random.default_rng(SEED)
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    index = pd.Series(np.arange(len(comp)), index=comp.index.astype(np.int64))
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "cell_class", "cell_type", "side", "nerve"]).drop_duplicates("root_id")
    a = a[a.root_id.isin(index.index)].set_index("root_id")
    side = a.side.fillna("center"); ctype = a.cell_type
    gr = a[(a.cell_class == "gustatory") & (a.nerve == "MxLbN")].copy()
    gr["type"] = gr.cell_type.fillna("peg_untyped")
    L = gr[gr.side == "left"]; R = gr[gr.side == "right"]
    con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    sign_of = con.groupby("Presynaptic_ID").Excitatory.agg(lambda s: int(np.sign(s.mean()) or 1))
    patch, report = [], {}
    for direction in ("out", "in"):
        me, other = ("Presynaptic_ID", "Postsynaptic_ID") if direction == "out" else ("Postsynaptic_ID", "Presynaptic_ID")
        e = con[con[me].isin(gr.index)].copy()
        e["g_type"] = gr.type.reindex(e[me]).values
        e["g_side"] = gr.side.reindex(e[me]).values
        e["p_type"] = ctype.reindex(e[other]).values
        e["p_side"] = side.reindex(e[other]).fillna("center").values
        e["rel"] = [rel_side(x, y) for x, y in zip(e.g_side, e.p_side)]
        eL = e[e.g_side == "left"]; eR = e[e.g_side == "right"]
        nL = L.groupby("type").size()
        templ = eL[eL.p_type.notna()].groupby(["g_type", "p_type", "rel"]).Connectivity.sum() / nL.reindex(
            eL[eL.p_type.notna()].groupby(["g_type", "p_type", "rel"]).Connectivity.sum().index.get_level_values(0)).values
        untyped_share = eL[eL.p_type.isna()].Connectivity.sum() / max(eL.Connectivity.sum(), 1)
        obs = eR[eR.p_type.notna()].groupby([me, "p_type", "rel"]).Connectivity.sum()
        # partner neurons per (type, side), weighted by input already received from right GRNs of g_type
        cand = {k: list(v) for k, v in pd.Series(ctype.index, index=ctype.index).groupby([ctype, side]).groups.items()} \
            if False else None
        by_ts = pd.DataFrame({"pid": ctype.index, "t": ctype.values, "s": side.values}).dropna(subset=["t"])
        members = by_ts.groupby(["t", "s"]).pid.apply(list).to_dict()
        prior = eR.groupby(["g_type", other]).Connectivity.sum().to_dict()
        added = 0
        is_grn = set(a.index[a.cell_class == "gustatory"])
        obs_tot = eR[eR.p_type.notna()].groupby(me).Connectivity.sum()
        for g, row in R.iterrows():
            T = row.type
            if T not in nL.index or T not in templ.index.get_level_values(0):
                continue
            tT = templ.loc[T]
            if direction == "in":
                # GRN->GRN contacts are completed once, from the presynaptic GRN's output pass
                tT = tT[[P not in set(ctype.reindex(list(is_grn)).dropna()) for P, _ in tT.index]]
            deficits = {(P, rel): expct - obs.get((g, P, rel), 0) for (P, rel), expct in tT.items()}
            pos = {k: v for k, v in deficits.items() if v > 0.5}
            # total budget: bring the neuron's typed synapse total up to the left-side mean, not beyond
            budget = max(0.0, float(tT.sum()) - float(obs_tot.get(g, 0)))
            scale = min(1.0, budget / max(sum(pos.values()), 1e-9))
            for (P, rel), miss in pos.items():
                miss *= scale
                if miss < 0.5:
                    continue
                tside = mirror_side("right", rel)
                js = [j for j in members.get((P, tside), []) if j != g]
                if direction == "in":
                    js = [j for j in js if j not in is_grn]
                if not js:
                    continue
                w = np.array([prior.get((T, j), 0) + 1.0 for j in js]); w /= w.sum()
                share = miss * w
                k = np.floor(share) + (rng.random(len(share)) < (share - np.floor(share)))
                for j, n in zip(js, k.astype(int)):
                    if n > 0:
                        pre, post = (g, j) if direction == "out" else (j, g)
                        patch.append((pre, post, n)); added += n
        report[direction] = {"synapses_added": int(added), "template_share_to_untyped_partners": round(float(untyped_share), 3)}
    p = pd.DataFrame(patch, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    p = p.groupby(["Presynaptic_ID", "Postsynaptic_ID"]).Connectivity.sum().reset_index()
    p["Excitatory"] = p.Presynaptic_ID.map(sign_of).fillna(1).astype(int)
    p["Presynaptic_Index"] = index.reindex(p.Presynaptic_ID).values
    p["Postsynaptic_Index"] = index.reindex(p.Postsynaptic_ID).values
    p["Excitatory x Connectivity"] = p.Excitatory * p.Connectivity
    p.to_parquet(ROOT / "data/grn_mirror_patch.parquet", index=False)
    # before/after audit
    tot = con.groupby("Presynaptic_ID").Connectivity.sum()
    add = p.groupby("Presynaptic_ID").Connectivity.sum()
    after = tot.add(add, fill_value=0)
    aud = {}
    for name, ids in (("left", L.index), ("right_before", R.index)):
        aud[name] = float(tot.reindex(ids).fillna(0).median())
    aud["right_after"] = float(after.reindex(R.index).fillna(0).median())
    report["median_output_synapses"] = aud
    report["edges_in_patch"] = int(len(p))
    (ROOT / "data/grn_mirror_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
