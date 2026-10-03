"""Phase 3 taste: build data/taste_neurons.csv and data/taste_organs.json.

Every FlyWire gustatory receptor neuron (GRN) is assigned to an organ, a side, a putative taste
modality and (for labellar bristles) a sensillum.

Organs (FlyWire 'nerve' + Tastekin et al. 2026 Cell types, flywire_annotations v3.2.0):
  labellar bristle GRNs  MxLbN, types LB1a-e, LB2a-c, LB3a-d, LB4a-b; modality from cell_sub_class
                         (measured: Tastekin et al. type -> receptor expression)
  labellar taste pegs    MxLbN, claw_tpGRN / dorsal_tpGRN / CB4146 (inner labellar surface)
  leg ascending GRNs     CV, LgAG1-9 (tarsal taste neurons that ascend to the brain)
  pharyngeal GRNs        PhN / aPhN (internal taste organs: LSO, VCSO, DCSO)

Modalities for pegs, leg and pharyngeal GRNs are not annotated. They are INFERRED here from
connectivity: each neuron gets the modality of the labellar bristle group whose downstream partner
profile (cosine similarity of output synapse vectors) it matches best; the similarity is stored and
anything below 0.3 is labelled 'unknown'. basis: approximate (connectivity inference).

Labellar sensilla (per side, Hiroi et al. 2002 Chem Senses 27:7): 11 L-type and 11 S-type bristles
with 4 GRNs each and 9 I-type with 2 GRNs; ~30 taste pegs with 1 GRN each (Falk et al. 1976). GRNs
are dealt to sensilla so each L-type holds one sugar, one water, one high-salt and one
sugar/low-salt neuron where available, and bitter neurons go to S- and I-type (the receptor
co-expression pattern of Hiroi et al. 2004 / Weiss et al. 2011). The assignment of a particular
FlyWire neuron to a particular bristle is NOT known. basis: approximate.
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODS = ["sugar", "sugar/low_salt", "water", "high_salt/heavy_metal", "bitter", "glutamate",
        "putative_attractive", "putative_aversive"]


def main():
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t",
                    usecols=["root_id", "cell_class", "cell_sub_class", "cell_type", "side", "nerve"],
                    low_memory=False).drop_duplicates("root_id")
    g = a[a.cell_class == "gustatory"].copy()
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    g = g[g.root_id.isin(comp.index)]
    g["organ"] = np.select(
        [(g.nerve == "MxLbN") & g.cell_type.str.startswith("LB", na=False),
         (g.nerve == "MxLbN"), g.nerve == "CV", g.nerve.isin(["PhN", "aPhN"])],
        ["labellum_bristle", "labellum_peg", "leg", "pharynx"], "unknown")
    g["modality"] = np.where(g.organ == "labellum_bristle", g.cell_sub_class, None)
    g["modality_basis"] = np.where(g.organ == "labellum_bristle", "measured (Tastekin et al. 2026 type)", "")
    # ---- infer modality of pegs / leg / pharyngeal GRNs from downstream partners
    con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    out = con[con.Presynaptic_ID.isin(g.root_id)]
    M = out.pivot_table(index="Presynaptic_ID", columns="Postsynaptic_ID", values="Connectivity",
                        aggfunc="sum", fill_value=0)
    M = M.div(np.sqrt((M ** 2).sum(1)), axis=0)
    lab = g[g.organ == "labellum_bristle"].set_index("root_id")
    proto = {m: M.reindex(lab.index[lab.modality == m]).dropna().mean() for m in MODS
             if (lab.modality == m).any()}
    P = pd.DataFrame(proto).T.fillna(0)
    P = P.div(np.sqrt((P ** 2).sum(1)), axis=0)
    sims = M.reindex(columns=P.columns, fill_value=0) @ P.T
    for i, r in g[g.organ != "labellum_bristle"].iterrows():
        if r.root_id in sims.index:
            s = sims.loc[r.root_id]
            best = s.idxmax()
            g.loc[i, "modality"] = best if s.max() >= 0.3 else "unknown"
            g.loc[i, "modality_basis"] = f"inferred from connectivity (cosine {s.max():.2f} to {best})"
        else:
            g.loc[i, "modality"] = "unknown"; g.loc[i, "modality_basis"] = "no outputs in FlyWire"
    # ---- Tastekin et al. 2026 (bioRxiv 10.1101/2025.08.25.671814): receptor matches and clusters override
    # the connectivity inference where they exist (data/tastekin2026_flywire_types.tsv for PhG types)
    tk = pd.read_csv(ROOT / "data/tastekin2026_flywire_types.tsv", sep="	", comment="#")
    phg = dict(zip(tk.root_id[tk["class"] == "pharyngeal_grn"], tk.type[tk["class"] == "pharyngeal_grn"]))
    g.loc[g.organ == "pharynx", "cell_type"] = g.root_id.map(phg).fillna(g.cell_type)
    TK = {  # type -> (modality, basis)
        "LgAG2": ("sugar", "literature: Gr61a-GAL4 match, appetitive cluster (Tastekin et al. 2026)"),
        "LgAG1": ("bitter", "literature: Gr33a-GAL4 match, aversive (Tastekin et al. 2026)"),
        "LgAG3": ("putative_aversive", "literature: aversive cluster (Tastekin et al. 2026)"),
        "LgAG5": ("putative_aversive", "literature: aversive cluster (Tastekin et al. 2026)"),
        "LgAG8": ("putative_aversive", "literature: aversive cluster (Tastekin et al. 2026)"),
        "LgAG4": ("putative_aversive", "literature: GNG016-sharing cluster, proposed aversive (Tastekin et al. 2026)"),
        "LgAG7": ("putative_aversive", "literature: GNG016-sharing cluster, proposed aversive (Tastekin et al. 2026)"),
        "LgAG9": ("putative_aversive", "literature: GNG016-sharing cluster, proposed aversive (Tastekin et al. 2026)"),
        "LgAG6": ("unknown", "literature: cluster of unknown relevance (Tastekin et al. 2026)"),
        "PhG1": ("sugar", "literature: Gr64e-GAL4 match, attractive cluster (Tastekin et al. 2026)"),
        "PhG10": ("putative_attractive", "literature: attractive cluster (Tastekin et al. 2026)"),
        "PhG3": ("water", "literature: proposed ppk28+ water GRN (Tastekin et al. 2026)"),
        "PhG4": ("water", "literature: proposed ppk28+ water GRN (Tastekin et al. 2026)"),
        "PhG2": ("putative_aversive", "literature: inhibits feeding via GNG055 (Tastekin et al. 2026)"),
    }
    for t in ["PhG5", "PhG6", "PhG7", "PhG9", "PhG11", "PhG12", "PhG13", "PhG14", "PhG15", "PhG16"]:
        TK[t] = ("putative_aversive", "literature: aversive cluster (Tastekin et al. 2026)")
    for t, (m, why) in TK.items():
        sel = g.cell_type == t
        g.loc[sel, "modality"] = m
        g.loc[sel, "modality_basis"] = why
    # ---- labellar sensilla per side, and dealing GRNs onto them
    sens = []
    for side in ("left", "right"):
        bri = g[(g.organ == "labellum_bristle") & (g.side == side)]
        pool = {m: list(bri.root_id[bri.modality == m]) for m in MODS}
        S = [{"id": f"lab_{side[0]}_L{k+1:02d}", "class": "L", "side": side, "grns": []} for k in range(11)] + \
            [{"id": f"lab_{side[0]}_I{k+1:02d}", "class": "I", "side": side, "grns": []} for k in range(9)] + \
            [{"id": f"lab_{side[0]}_S{k+1:02d}", "class": "S", "side": side, "grns": []} for k in range(11)]
        want = {"L": ["sugar", "water", "high_salt/heavy_metal", "sugar/low_salt"],
                "I": ["sugar/low_salt", "bitter"],
                "S": ["sugar", "bitter", "high_salt/heavy_metal", "glutamate"]}
        for s_ in S:
            for m in want[s_["class"]]:
                if pool[m]:
                    s_["grns"].append(str(pool[m].pop(0)))
        # leftovers (more neurons of a modality than slots): spread over sensilla with < 4 GRNs
        cap = {"L": 4, "I": 2, "S": 4}
        left = [r for m in MODS for r in pool[m]]
        for s_ in S * 3:
            while left and len(s_["grns"]) < cap[s_["class"]]:
                s_["grns"].append(str(left.pop(0)))
        for r in left:      # still unplaced: extra GRNs on S-type bristles (FlyWire has a few more)
            S[-1 - (hash(r) % 11)]["grns"].append(str(r))
        pegs = list(g.root_id[(g.organ == "labellum_peg") & (g.side == side)])
        for k, r in enumerate(pegs):
            S.append({"id": f"peg_{side[0]}_{k+1:02d}", "class": "peg", "side": side, "grns": [str(r)]})
        sens += S
    sid = {r: s_["id"] for s_ in sens for r in s_["grns"]}
    g["sensillum"] = g.root_id.astype(str).map(sid)
    g["index"] = pd.Series(np.arange(len(comp)), index=comp.index.astype(np.int64)).reindex(g.root_id).values
    # leg sensors: LgAG neurons of a side are dealt round-robin over that side's 15 tarsal segments
    # (FlyWire does not say which leg or segment an ascending leg GRN comes from: basis approximate)
    for side, sl in (("left", "l"), ("right", "r")):
        # leg of origin from Tastekin et al. 2026 (male CNS VNC entry): LgAG1/2 all legs, LgAG3/4/8
        # mid+hind legs, LgAG5/6/7/9 forelegs. Segment within the tarsus: not known (approximate).
        LEGS = {"LgAG1": "fmh", "LgAG2": "fmh", "LgAG3": "mh", "LgAG4": "mh", "LgAG8": "mh",
                "LgAG5": "f", "LgAG6": "f", "LgAG7": "f", "LgAG9": "f"}
        for t, legset in LEGS.items():
            slots = [f"leg_{sl}{x}_tarsus{k}" for k in (5, 4, 3, 2, 1) for x in legset]
            ids = g.index[(g.organ == "leg") & (g.side == side) & (g.cell_type == t)]
            for j, i in enumerate(ids):
                g.loc[i, "sensillum"] = slots[j % len(slots)]
    g.loc[g.organ == "pharynx", "sensillum"] = "pharynx"
    # ---- leg taste neurons that stay in the nerve cord (BANC, tools/build_vnc.py): receptor-labelled
    # (BANC 'function', e.g. Gr5a/Gr64f sugar, ppk23/ppk25 pheromone, Ir52 contact pheromone)
    vp = ROOT / "data/vnc/vnc_neurons.csv"
    if vp.exists():
        vn = pd.read_csv(vp)
        lg = vn[(vn.cell_class == "taste_bristle_gustatory_neuron") & vn.body_part.isin(["front_leg", "middle_leg", "hind_leg"])].copy()
        def vmod(f):
            f = str(f).lower()
            if "bitter" in f or "gr66a" in f or "gr33a" in f: return "bitter"
            if "low_salt" in f and "sugar" in f: return "sugar/low_salt"
            if "sugar" in f or "gr5a" in f or "gr64f" in f or "gr61a" in f: return "sugar"
            if "ppk28" in f: return "water"
            if "pheromone" in f or "ppk25" in f or "ppk23" in f: return "pheromone"
            return "unknown"
        rows = []
        legch = {"front_leg": "f", "middle_leg": "m", "hind_leg": "h"}
        for (bp, sd), d in lg.groupby(["body_part", "side"]):
            slots = [f"leg_{str(sd)[0]}{legch[bp]}_tarsus{k}" for k in (5, 4, 3, 2, 1)]
            for j, (_, r) in enumerate(d.iterrows()):
                rows.append({"root_id": int(r.bid), "cell_class": "gustatory", "cell_sub_class": r.sub_class,
                             "cell_type": r.cell_type, "side": r.side, "nerve": "leg nerve (BANC)", "organ": "leg_vnc",
                             "modality": vmod(r.function),
                             "modality_basis": f"measured: BANC receptor annotation '{r.function}'",
                             "sensillum": slots[j % len(slots)], "index": int(r.model_index)})
        g = pd.concat([g, pd.DataFrame(rows)], ignore_index=True)
        print(f"VNC leg taste neurons added: {len(rows)}", pd.DataFrame(rows).modality.value_counts().to_dict())
    g.to_csv(ROOT / "data/taste_neurons.csv", index=False)
    summ = g.groupby(["organ", "modality"]).size()
    print(summ.to_string())
    print("\nleg/pharynx/peg modality inference (median cosine):",
          g[g.organ != "labellum_bristle"].modality_basis.str.extract(r"cosine ([\d.]+)")[0].astype(float).median())
    (ROOT / "data/taste_organs.json").write_text(json.dumps({
        "note": __doc__.strip().splitlines()[0], "sensilla": sens,
        # Shiu et al. 2024 Fig. 1 benchmark: 21 right labellar sugar GRNs (their figures.ipynb)
        # (strings: Godot's JSON parser turns 18-digit IDs into doubles and loses precision)
        "shiu_sugar_R": [str(x) for x in [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
                         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
                         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
                         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
                         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
                         720575940611875570]],
        "basis": {"bristle_modality": "measured (Tastekin et al. 2026 types)",
                  "peg_leg_pharynx_modality": "approximate (connectivity inference)",
                  "neuron_to_sensillum": "approximate (dealt by co-expression rules)"}}, indent=1))
    print(f"\n{len(sens)} labellar sensilla written; {sum(len(s['grns']) for s in sens)} GRNs placed")


if __name__ == "__main__":
    main()
