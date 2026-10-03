"""Phase 4: attach the ventral nerve cord (VNC) from BANC to the FlyWire brain.

BANC (Brain And Nerve Cord; Bates et al. 2025; Codex release 888, data/raw/banc/) is a female CNS
reconstructed in one animal. FlyWire (FAFB v783) has no VNC. The two are joined the way the real
animal is: through the neck.
  - Descending neurons (DN, brain -> VNC): every FlyWire DN is matched to a BANC DN by cell type
    (primary or alternative name) and side; when a (type, side) group has several neurons they are
    paired by similarity of their BRAIN inputs. The FlyWire DN then gets its BANC twin's VNC
    synapses (outputs and inputs in VNC neuropils).
  - Ascending (AN) and sensory-ascending (SA) neurons (VNC -> brain): names differ between datasets
    (FlyWire AN_* vs MANC-style AN07B041), so FlyWire ANs/SAs are matched to BANC ANs/SAs by the
    similarity of their BRAIN connectivity (partner cell types, both datasets use FlyWire type
    names for 89 % of central-brain neurons), optimal assignment. The match method is validated on
    DNs, whose names are known (accuracy printed). Accepted pairs give the FlyWire neuron its BANC
    twin's VNC synapses.
  - All other BANC neurons whose synapses are mainly in VNC neuropils (VNC interneurons, leg/wing/
    haltere/neck motor neurons, leg/wing/haltere sensory neurons, unmatched ANs) are ADDED as new
    neurons with their BANC VNC connectivity.
Only synapses in VNC neuropils are taken from BANC; the brain stays FlyWire. Sign: GABA/glutamate
inhibitory, otherwise excitatory (as in Shiu et al. 2024). basis: measured (BANC), joined by
matching (approximate where similarity-based; see data/vnc/vnc_matching_report.json).

Outputs: data/vnc/vnc_neurons.csv, data/vnc/vnc_edges.parquet, data/vnc/vnc_matching_report.json
"""
import json, pathlib, re
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/vnc"
VNC_NP = re.compile(r"^(VNC_|IntTct|LTct|HTct|NTct|WTct|ABDNM|ANm|LegNp)", re.I)
INHIB = {"gaba", "glutamate", "GABA", "GLUT", "Glutamate"}


def norm_rows(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ M


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------------- FlyWire side
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    fw_ids = comp.index.to_numpy(np.int64); N_FW = len(fw_ids)
    fw_index = pd.Series(np.arange(N_FW), index=fw_ids)
    fa = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                     usecols=["root_id", "super_class", "cell_type", "side"]).drop_duplicates("root_id").set_index("root_id").reindex(fw_ids)
    fcon = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                           columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    fw_sign = fcon.groupby("Presynaptic_ID").Excitatory.agg(lambda s: 1 if s.mean() >= 0 else -1)
    fkey = (fa.cell_type.fillna("?") + "|" + fa.side.fillna("?")).to_dict()
    # ---------------- BANC side
    bn = pd.read_csv(ROOT / "data/raw/banc/neurons.csv.gz").rename(columns={"Root ID": "bid"})
    fw_types = set(fa.cell_type.dropna())

    def fw_name(row):
        cands = [row["Primary Cell Type"]] + [x.strip() for x in str(row["Alternative Cell Type(s)"]).split(",") if x.strip()]
        for c in cands:
            if c in fw_types:
                return c
        return row["Primary Cell Type"] if isinstance(row["Primary Cell Type"], str) else "?"
    bn["fw_type"] = bn.apply(fw_name, axis=1)
    bn["side"] = bn["Soma side"].fillna("?")
    bkey = (bn.fw_type + "|" + bn.side).set_axis(bn.bid).to_dict()
    bc = pd.read_csv(ROOT / "data/raw/banc/connections_princeton.csv.gz", usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    in_vnc = bc.neuropil.fillna("").str.match(VNC_NP)
    bbrain = bc[~in_vnc].groupby(["pre_root_id", "post_root_id"]).syn_count.sum().reset_index()
    bvnc = bc[in_vnc].groupby(["pre_root_id", "post_root_id"]).syn_count.sum().reset_index()
    vnc_syn = pd.concat([bvnc.groupby("pre_root_id").syn_count.sum(), bvnc.groupby("post_root_id").syn_count.sum()]).groupby(level=0).sum()
    brain_syn = pd.concat([bbrain.groupby("pre_root_id").syn_count.sum(), bbrain.groupby("post_root_id").syn_count.sum()]).groupby(level=0).sum()
    bn["vnc_frac"] = bn.bid.map(vnc_syn).fillna(0) / (bn.bid.map(vnc_syn).fillna(0) + bn.bid.map(brain_syn).fillna(0)).replace(0, np.nan)

    # ---------------- brain-connectivity profiles (partner type|side), for matching
    def profiles(edges, pre, post, w, ids, key):
        vocab = {}
        rows, cols, vals = [], [], []
        idpos = {x: i for i, x in enumerate(ids)}
        for direction, me, other in (("o", pre, post), ("i", post, pre)):
            e = edges[edges[me].isin(idpos)]
            ks = [direction + key.get(x, "?|?") for x in e[other]]
            for r, k, v in zip(e[me].map(idpos), ks, e[w]):
                if k.endswith("?|?"):
                    continue
                c = vocab.setdefault(k, len(vocab))
                rows.append(r); cols.append(c); vals.append(v)
        return rows, cols, vals, vocab

    def similarity(f_ids, b_ids):
        fe = fcon[fcon.Presynaptic_ID.isin(f_ids) | fcon.Postsynaptic_ID.isin(f_ids)]
        be = bbrain[bbrain.pre_root_id.isin(b_ids) | bbrain.post_root_id.isin(b_ids)]
        fr, fc, fv, voc = profiles(fe, "Presynaptic_ID", "Postsynaptic_ID", "Connectivity", list(f_ids), fkey)
        br, bc_, bv, voc2 = profiles(be, "pre_root_id", "post_root_id", "syn_count", list(b_ids), bkey)
        allk = {k: i for i, k in enumerate(set(voc) | set(voc2))}
        inv1 = {v: allk[k] for k, v in voc.items()}; inv2 = {v: allk[k] for k, v in voc2.items()}
        F = sp.csr_matrix((fv, (fr, [inv1[c] for c in fc])), shape=(len(f_ids), len(allk)))
        B = sp.csr_matrix((bv, (br, [inv2[c] for c in bc_])), shape=(len(b_ids), len(allk)))
        return (norm_rows(F) @ norm_rows(B).T).toarray()

    report = {}
    # ---------------- DNs: by name+side, pairs within groups by brain-input similarity
    fdn = fa[fa.super_class == "descending"]
    bdn = bn[bn["Super Class"] == "descending"]
    dn_pairs = {}
    for (t, s), g in fdn.groupby([fdn.cell_type.fillna("?"), fdn.side.fillna("?")]):
        cand = bdn[(bdn.fw_type == t) & (bdn.side == s)]
        if len(cand) == 0:
            continue
        S = similarity(list(g.index), list(cand.bid))
        r, c = linear_sum_assignment(-S)
        for i, j in zip(r, c):
            dn_pairs[g.index[i]] = (cand.bid.iloc[j], float(S[i, j]))
    report["DN"] = {"flywire": int(len(fdn)), "matched_by_name": len(dn_pairs)}
    # validation of the similarity matcher on DNs (names hidden): accuracy of top-1 type
    vf = [f for f in dn_pairs][:600]; vb = list(bdn.bid)
    S = similarity(vf, vb)
    top = [bkey.get(vb[k], "?").split("|")[0] for k in S.argmax(1)]
    acc = np.mean([top[i] == fa.cell_type[f] for i, f in enumerate(vf)])
    report["DN"]["similarity_matcher_top1_type_accuracy"] = round(float(acc), 3)
    print(f"DN: {len(dn_pairs)}/{len(fdn)} matched by name; similarity-only matcher recovers the right type for {acc:.0%}")
    # ---------------- ANs / SAs: by brain-connectivity similarity
    an_pairs = {}
    for fsc, bsc in (("ascending", ["ascending"]), ("sensory_ascending", ["sensory_ascending"])):
        f_ids = list(fa.index[fa.super_class == fsc]); b_ids = list(bn.bid[bn["Super Class"].isin(bsc)])
        S = similarity(f_ids, b_ids)
        # names, where they do exist (e.g. LgAG1), override
        r, c = linear_sum_assignment(-S)
        acc_thr = 0.4
        kept = [(f_ids[i], b_ids[j], float(S[i, j])) for i, j in zip(r, c) if S[i, j] >= acc_thr]
        # type-consistent completion: cell types come as left/right homolog sets. From the confident matches,
        # learn FlyWire type -> BANC type by majority vote (>= 50% of that type's matches); then pair the
        # still-unmatched members of each type with still-unmatched members of its BANC counterpart type, by
        # brain-connectivity similarity (Hungarian within the group). Recovers e.g. the left Dandelion
        # (AN_GNG_68 = AN13B002; Tastekin et al. 2026) whose partner was below the global threshold.
        n_conf = len(kept)
        btype = bn.set_index("bid")["Primary Cell Type"]
        used_b = {b for _, b, _ in kept}; used_f = {f for f, _, _ in kept}
        votes = pd.DataFrame([(fa.cell_type.get(f), btype.get(b)) for f, b, _ in kept], columns=["ft", "bt"]).dropna()
        cross = {}
        for ft, g in votes.groupby("ft"):
            top = g.bt.value_counts()
            if top.iloc[0] / len(g) >= 0.5:
                cross[ft] = top.index[0]
        fi = {x: i for i, x in enumerate(f_ids)}; bi = {x: j for j, x in enumerate(b_ids)}
        b_by_type = pd.Series(b_ids).groupby(pd.Series(b_ids).map(btype)).apply(list).to_dict()
        added = 0
        for ft, bt in cross.items():
            fs = [f for f in f_ids if fa.cell_type.get(f) == ft and f not in used_f]
            bs = [b for b in b_by_type.get(bt, []) if b not in used_b]
            if not fs or not bs:
                continue
            sub = S[np.ix_([fi[f] for f in fs], [bi[b] for b in bs])]
            rr, cc = linear_sum_assignment(-sub)
            for i, j in zip(rr, cc):
                if sub[i, j] > 0:
                    kept.append((fs[i], bs[j], float(sub[i, j]))); used_f.add(fs[i]); used_b.add(bs[j]); added += 1
        for f, b, s in kept:
            an_pairs[f] = (b, s)
        named = [(f, b, s) for f, b, s in kept if fa.cell_type[f] == bn.set_index("bid").fw_type.get(b)]
        report[fsc] = {"flywire": len(f_ids), "banc": len(b_ids), "matched": len(kept),
                       "matched_confident": n_conf, "added_by_type_consistency": added, "type_crosswalk_size": len(cross),
                       "median_similarity": round(float(np.median([k[2] for k in kept])) if kept else 0, 3),
                       "pairs_with_identical_type_name": len(named)}
        print(f"{fsc}: {len(kept)}/{len(f_ids)} FlyWire neurons matched (cosine >= {acc_thr}, median "
              f"{report[fsc]['median_similarity']}); {len(named)} of them also share a type name")
    # ---------------- VNC-resident BANC neurons -> new model neurons
    bridged_b = {b: f for f, (b, _) in {**dn_pairs, **an_pairs}.items()}
    res = bn[(bn.vnc_frac >= 0.5) & ~bn.bid.isin(bridged_b) & (bn["Super Class"] != "descending")].copy()
    res = res[res["Super Class"].fillna("") != "glia"]
    res["model_index"] = N_FW + np.arange(len(res))
    idx_of = dict(zip(res.bid, res.model_index))
    for b, f in bridged_b.items():
        idx_of[b] = int(fw_index[f])
    # ---------------- edges in VNC neuropils among (resident + bridged)
    e = bvnc[bvnc.pre_root_id.isin(idx_of) & bvnc.post_root_id.isin(idx_of)].copy()
    e["Presynaptic_Index"] = e.pre_root_id.map(idx_of); e["Postsynaptic_Index"] = e.post_root_id.map(idx_of)
    # Dale's principle, as in build_connectome.py: ACh +, GABA/Glu -, monoamines 0 (no fast effect)
    def nsign(x):
        x = str(x).lower()
        if "acetylcholine" in x or x in ("ach",): return 1
        if "gaba" in x or "glut" in x or "hist" in x: return -1     # histamine: HisCl/Ort chloride channels
        if x in ("da", "ser", "oct", "tyr") or any(k in x for k in ("dopamine", "serotonin", "octopamine", "tyramine")): return 0
        return 1
    nt = bn.set_index("bid")["Predicted NT type"].fillna("")
    sign_b = np.array([nsign(x) for x in nt.reindex(e.pre_root_id).fillna("").values])
    # bridged FlyWire neurons keep their FlyWire neuron-level transmitter (known_nt, else top_nt)
    fa_nt = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="	", low_memory=False,
                        usecols=["root_id", "top_nt", "known_nt"]).drop_duplicates("root_id").set_index("root_id")
    fnt = fa_nt.known_nt.fillna(fa_nt.top_nt).fillna("")
    fsign = np.array([nsign(fnt.get(bridged_b[p], "")) if p in bridged_b else 99 for p in e.pre_root_id])
    e["Excitatory"] = np.where(fsign != 99, fsign, sign_b)
    e["Connectivity"] = e.syn_count.astype(int)
    e["Excitatory x Connectivity"] = e.Excitatory * e.Connectivity
    e[["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory", "Excitatory x Connectivity",
       "pre_root_id", "post_root_id"]].to_parquet(OUT / "vnc_edges.parquet", index=False)
    res = res.rename(columns={"Super Class": "super_class", "Class": "cell_class", "Sub Class": "sub_class",
                              "Primary Cell Type": "cell_type", "Body Part": "body_part", "Predicted NT type": "nt",
                              "Hemilineage": "hemilineage", "Nerve": "nerve", "Function": "function"})
    res[["model_index", "bid", "super_class", "cell_class", "sub_class", "cell_type", "fw_type", "side", "body_part",
         "nerve", "nt", "hemilineage", "function", "vnc_frac"]].to_csv(OUT / "vnc_neurons.csv", index=False)
    bridge = pd.DataFrame([(f, b, s, "DN") for f, (b, s) in dn_pairs.items()] +
                          [(f, b, s, "AN/SA") for f, (b, s) in an_pairs.items()],
                          columns=["flywire_id", "banc_id", "similarity", "kind"])
    bridge.to_csv(OUT / "vnc_bridge.csv", index=False)
    report["vnc_resident_neurons"] = int(len(res))
    report["vnc_edges"] = int(len(e)); report["vnc_synapses"] = int(e.Connectivity.sum())
    report["resident_by_class"] = res.super_class.value_counts().to_dict()
    (OUT / "vnc_matching_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "resident_by_class"}, indent=1))
    print("resident:", report["resident_by_class"])


if __name__ == "__main__":
    main()
