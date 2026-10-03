"""Phase 2: build data/ommatidia.json, one record per real ommatidium (facet), with its
optical axis and the FlyWire v783 neurons of the visual column it feeds.

Sources (all in data/raw/):
  eyemap_T4/          Zhao et al. 2025, Nature 643:1149, "Eye structure shapes neuron function in
                      Drosophila motion vision" (github.com/reiserlab/eyemap_T4, GPL-3).
                      microCT/20240701.RData: micro-CT of one female head, 1709 lenses with
                      positions (um) and optical axes (cone->lens unit vectors, hex-smoothed),
                      head frame +x anterior, +y left, +z dorsal.
                      eyemap.RData: right-eye lens <-> FAFB Mi1 (medulla column) matching, and the
                      lens hex grid lens_ixy (Zhao's i,j coordinates).
                      Mi1_ind.RData: Mi1 columns in the dorsal rim area (DRA).
  column_assignment.csv.gz   FlyWire Codex (Matsliah et al. 2024, Nature 634:166): hex (p,q) column
                      of every columnar visual neuron, both optic lobes.
  visual_neuron_types.csv.gz FlyWire Codex visual types (for R1-6, which have no column).
  Connectivity_783.parquet   FlyWire v783 connections (Shiu et al. 2024 export).

Key link, verified here: Codex (p, q) == Zhao (i - 1, j). Checked two independent ways against
Zhao's FAFB Mi1 grid: lattice occupancy (770/779) and photoreceptor-less edge columns (40/56,
next-best transform 26/56). See check_transform().

Basis labels per facet (field "basis"):
  direction  measured      right and left eye: micro-CT (a different fly than FlyWire's FAFB brain
                           and than the NeuroMechFly body; eyes of same-size females, Zhao et al.)
  column     measured      right eye: Zhao's lens<->Mi1 matching of the FAFB eye + Codex columns
             approximate   right-eye lenses Zhao left unmatched (eye edge): same hex coordinate rule
             approximate   left eye: each left lens takes the (p,q) of the nearest right-eye axis
                           after mirroring (assumes bilateral symmetry of the lattice)
  R1-6       measured      each R1-6 is placed in the column of its strongest L1+L2+L3 targets
                           (its lamina cartridge). By neural superposition that cartridge collects
                           light along that column's axis, which is what drives it here.
  subtype    approximate   yellow/pale: FlyWire has no p/y labels; drawn at random, 70% yellow,
                           30% pale (real ratio; the real pattern is random in every fly).
             measured      DRA (right eye): Zhao's DRA Mi1 columns in FAFB.
             approximate   DRA (left eye): mirrored from right.
"""
import json, pathlib, collections
import numpy as np, pandas as pd, rdata, warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
EM = RAW / "eyemap_T4"
SEED = 783  # fixed seed so the yellow/pale draw is reproducible
COLUMN_TYPES = ["R7", "R8", "L1", "L2", "L3", "L4", "L5", "C2", "C3", "T1", "T2", "T2a", "T3",
                "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Tm20", "Tm21",
                "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def rd(path):
    return rdata.read_rda(str(path))


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))


def load_microct():
    u = rd(EM / "microCT/20240701.RData")
    order = np.argsort(np.asarray(u["i_match"]).astype(int) - 1)  # cone order -> lens order
    left = np.asarray(u["ind_left_lens"]).astype(bool)
    lens = np.asarray(u["lens"])
    axis = np.asarray(u["ucl_rot_sm"])[order]
    return {"right": (lens[~left], axis[~left]), "left": (lens[left], axis[left])}


def check_transform(ij, codex_right):
    """Return (occupancy, noPR hits) for Codex (p,q) = Zhao (i-1, j)."""
    allpq = set(map(tuple, codex_right[["p", "q"]].values))
    prpq = set(map(tuple, codex_right[codex_right.type.isin(["R7", "R8"])][["p", "q"]].values))
    nopr = np.asarray(rd(EM / "Mi1_ind.RData")["Mi1_ind_noPR"]).astype(int)
    occ = sum((i - 1, j) in allpq for i, j in ij.values())
    hits = sum((ij[k][0] - 1, ij[k][1]) in (allpq - prpq) for k in nopr if k in ij)
    return occ, len(ij), hits, len(nopr)


def r16_cartridges(codex, vtypes):
    con = pd.read_parquet(RAW / "Connectivity_783.parquet",
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    r16 = vtypes[vtypes.type == "R1-6"][["root_id", "side"]]
    lam = codex[codex.type.isin(["L1", "L2", "L3"])][["root_id", "hemisphere", "p", "q"]]
    m = con[con.Presynaptic_ID.isin(r16.root_id)].merge(
        lam, left_on="Postsynaptic_ID", right_on="root_id")
    w = m.groupby(["Presynaptic_ID", "hemisphere", "p", "q"]).Connectivity.sum().reset_index()
    tot = w.groupby("Presynaptic_ID").Connectivity.transform("sum")
    w["frac"] = w.Connectivity / tot
    best = w.sort_values("Connectivity").groupby("Presynaptic_ID").tail(1)
    best = best.merge(r16, left_on="Presynaptic_ID", right_on="root_id")
    return best, len(r16)


def main():
    codex = pd.read_csv(RAW / "column_assignment.csv.gz")
    vtypes = pd.read_csv(RAW / "visual_neuron_types.csv.gz")
    em = rd(EM / "eyemap.RData")
    med_ixy = np.asarray(rd(EM / "med_ixy.RData")["med_ixy"]).astype(int)
    ij_mi1 = {m[0]: (m[1], m[2]) for m in med_ixy}
    occ, n_mi1, hits, n_nopr = check_transform(ij_mi1, codex[codex.hemisphere == "right"])
    print(f"transform check: occupancy {occ}/{n_mi1}, noPR edge columns {hits}/{n_nopr}")

    eyemap = np.asarray(em["eyemap"]).astype(int)               # [Mi1 idx, lens idx] 1-based
    lens_ixy = np.asarray(em["lens_ixy"]).astype(int)           # [lens idx, i, j]
    lens_hex = {r[0]: (r[1], r[2]) for r in lens_ixy}
    matched_lens = set(eyemap[:, 1])
    dra_mi1 = set(np.asarray(rd(EM / "Mi1_ind.RData")["Mi1_ind_DRA"]).astype(int))
    dra_lens = {l for m, l in eyemap if m in dra_mi1}
    ct = load_microct()

    # neurons per (hemisphere, p, q)
    neur = collections.defaultdict(lambda: collections.defaultdict(list))
    colid = {}
    for rid, hemi, typ, cid, p, q in codex[["root_id", "hemisphere", "type", "column_id", "p", "q"]].values:
        neur[(hemi, p, q)][typ].append(str(rid))
        colid[(hemi, p, q)] = int(cid)
    r16, n_r16 = r16_cartridges(codex, vtypes)
    for rid, hemi, p, q, frac in r16[["Presynaptic_ID", "hemisphere", "p", "q", "frac"]].values:
        neur[(hemi, p, q)]["R1-6"].append(str(rid))
    print(f"R1-6 placed in cartridges: {len(r16)}/{n_r16} "
          f"(median share of L1-3 output to that cartridge {np.median(r16.frac):.2f})")

    rng = np.random.default_rng(SEED)
    facets = []
    # ---- right eye
    lens_r, ax_r = ct["right"]
    for li in range(1, len(lens_r) + 1):
        if li not in lens_hex:
            continue
        i, j = lens_hex[li]
        p, q = i - 1, j
        key = ("right", p, q)
        facets.append({
            "side": "right", "lens_index": li, "hex_zhao": [i, j], "pq": [p, q],
            "column_id": colid.get(key), "lens_um": lens_r[li - 1].round(2).tolist(),
            "dir": (ax_r[li - 1] / np.linalg.norm(ax_r[li - 1])).round(5).tolist(),
            "dra": li in dra_lens,
            "basis": {"direction": "measured",
                      "column": "measured" if li in matched_lens else "approximate",
                      "dra": "measured"},
        })
    # ---- left eye: nearest mirrored right axis, greedy unique
    lens_l, ax_l = ct["left"]
    R = np.array([f["dir"] for f in facets]); R_m = R * [1, -1, 1]
    A = ax_l / np.linalg.norm(ax_l, axis=1, keepdims=True)
    d = angle_deg(A[:, None, :], R_m[None, :, :])
    pairs = sorted(((d[a, b], a, b) for a in range(len(A)) for b in np.argsort(d[a])[:6]))
    used_a, used_b, match = set(), set(), {}
    for dist, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b); match[a] = (b, dist)
    mism = [v[1] for v in match.values()]
    print(f"left eye: {len(match)}/{len(A)} lenses matched to mirrored right lattice, "
          f"median axis mismatch {np.median(mism):.2f} deg")
    for a in range(len(A)):
        rec = {"side": "left", "lens_index": a + 1, "lens_um": lens_l[a].round(2).tolist(),
               "dir": A[a].round(5).tolist(), "hex_zhao": None, "pq": None, "column_id": None,
               "dra": False,
               "basis": {"direction": "measured", "column": "none", "dra": "approximate"}}
        if a in match:
            b = facets[match[a][0]]
            p, q = b["pq"]
            rec.update(pq=[p, q], hex_zhao=b["hex_zhao"], column_id=colid.get(("left", p, q)),
                       dra=b["dra"], mirror_match_deg=round(float(match[a][1]), 2))
            rec["basis"]["column"] = "approximate"
        facets.append(rec)

    # ---- per-facet neurons, subtype, acceptance angle, neighbour angle
    for k, f in enumerate(facets):
        f["id"] = k
        cell = neur.get((f["side"], *f["pq"])) if f["pq"] else None
        f["neurons"] = {t: cell[t] for t in COLUMN_TYPES + ["R1-6"] if cell and cell.get(t)} if cell else {}
        f["subtype"] = "DRA" if f["dra"] else ("pale" if rng.random() < 0.30 else "yellow")
        f["basis"]["subtype"] = f["basis"]["dra"] if f["dra"] else "approximate"
        # Acceptance angle ~5 deg (Gonzalez-Bellido et al. 2011 PNAS; literature)
        f["acceptance_deg"] = 5.0
        f["basis"]["acceptance"] = "literature"
    for side in ("right", "left"):
        idx = [f["id"] for f in facets if f["side"] == side]
        D = np.array([facets[i]["dir"] for i in idx])
        ang = angle_deg(D[:, None, :], D[None, :, :]); np.fill_diagonal(ang, 999)
        nn = np.sort(ang, axis=1)[:, :6]
        for n, i in enumerate(idx):
            facets[i]["interommatidial_deg"] = round(float(np.median(nn[n])), 2)
        with_cols = sum(1 for i in idx if facets[i]["neurons"])
        n_ids = sum(len(v) for i in idx for v in facets[i]["neurons"].values())
        print(f"{side}: {len(idx)} facets, {with_cols} with FlyWire column neurons ({n_ids} neuron IDs), "
              f"interommatidial angle median {np.median(nn):.2f} deg, "
              f"DRA {sum(facets[i]['dra'] for i in idx)}, "
              f"pale {sum(facets[i]['subtype']=='pale' for i in idx)}")

    out = {
        "units": {"lens_um": "micrometres, micro-CT head frame (+x anterior, +y left, +z dorsal)",
                  "dir": "unit optical axis, same head frame"},
        "sources": {
            "eye": "Zhao et al. 2025 Nature 643:1149, reiserlab/eyemap_T4 @ "
                   + (EM / "SOURCE_COMMIT.txt").read_text().strip(),
            "columns": "FlyWire Codex v783 column_assignment (Matsliah et al. 2024 Nature 634:166)",
            "connectome": "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024)",
        },
        "transform": "Codex (p,q) = Zhao (i-1, j)",
        "transform_check": {"occupancy": [occ, n_mi1], "noPR_edge": [hits, n_nopr]},
        "facets": facets,
    }
    (ROOT / "data/ommatidia.json").write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote data/ommatidia.json,", len(facets), "facets")


if __name__ == "__main__":
    main()
