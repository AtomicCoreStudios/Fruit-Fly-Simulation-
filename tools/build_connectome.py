"""
Build the connectome binary that the Godot simulation loads.

Two modes:

  python tools/build_connectome.py synthetic [--scale 1.0] [--seed 1]
      Generates a FlyWire-scale (139,255 neuron) *stand-in* brain whose
      neuropils, cell populations and pathways follow the known fly circuit
      layout (optic lobes, antennal lobes, mushroom bodies, lateral horn,
      central complex, SEZ, AMMC, descending neurons). Wiring statistics are
      hand-designed, NOT the real synapse-level connectome.

  python tools/build_connectome.py flywire --raw data/raw
      Uses the real FlyWire v783 connectome in the format published with
      Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model):
        data/raw/Completeness_783.csv
        data/raw/Connectivity_783.parquet   (needs `pip install pyarrow pandas`)
      Optional FlyWire Codex annotation exports (improve positions/IO mapping):
        data/raw/classification.csv   (root_id, super_class, class, sub_class, cell_type, side ...)
        data/raw/coordinates.csv      (root_id, position)

Output (read by scripts/connectome.gd):
  data/connectome.bin   binary arrays
  data/connectome.json  region / group / parameter metadata
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data")
TEX_W = 1024  # activity texture width used by the viewer
FIXED = 1000.0  # weights stored as int32 micro-volts

SECTORS = 6  # azimuthal visual sectors per eye
COMPASS = 8  # central-complex heading wedges

# ---------------------------------------------------------------------------
# Regions (neuropils) - name, colour, ellipsoid centre (um), radii (um)
# x: +left / -right, y: dorsal, z: anterior
# ---------------------------------------------------------------------------
REGIONS = [
    ("Optic lobe L", "#3a7bd5", (230, 0, -10), (90, 140, 95)),
    ("Optic lobe R", "#3a7bd5", (-230, 0, -10), (90, 140, 95)),
    ("Antennal lobe L", "#e8a33d", (50, -45, 85), (32, 30, 28)),
    ("Antennal lobe R", "#e8a33d", (-50, -45, 85), (32, 30, 28)),
    ("Mushroom body L", "#c04fd0", (75, 45, -10), (45, 55, 80)),
    ("Mushroom body R", "#c04fd0", (-75, 45, -10), (45, 55, 80)),
    ("Lateral horn L", "#4fd08a", (120, 40, -45), (35, 35, 30)),
    ("Lateral horn R", "#4fd08a", (-120, 40, -45), (35, 35, 30)),
    ("Central complex", "#f25c54", (0, 35, -5), (65, 22, 28)),
    ("SEZ / gustatory", "#f2d05c", (0, -95, 25), (65, 40, 45)),
    ("AMMC / JO L", "#5cc8f2", (75, -55, 55), (28, 25, 25)),
    ("AMMC / JO R", "#5cc8f2", (-75, -55, 55), (28, 25, 25)),
    ("Central brain", "#9a9a9a", (0, 5, -10), (165, 115, 95)),
    ("Descending", "#ffffff", (0, -150, -50), (18, 60, 18)),
    ("Ascending", "#b0ffb0", (0, -150, -70), (18, 60, 18)),
    # ventral nerve cord (BANC, tools/build_vnc.py); viewer positions are schematic (below/behind brain)
    ("VNC interneurons", "#ff9de2", (0, -330, -380), (70, 60, 260)),
    ("VNC sensory", "#9de2ff", (0, -330, -380), (90, 70, 290)),
    ("VNC motor", "#ffd59d", (0, -330, -380), (100, 75, 300)),
]
RID = {r[0]: i for i, r in enumerate(REGIONS)}


class Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.pops = {}  # name -> (start, count, sign)
        self.n = 0
        self.region = []
        self.popid = []
        self.pop_meta = []
        self.group = []
        self.pos = []
        self.groups = [{"name": "none", "role": "none"}]
        self.gid = {"none": 0}
        self.edges_pre = []
        self.edges_post = []
        self.edges_w = []

    # ---- groups: sensory inputs / motor readouts / background drives ----
    def group_id(self, name, role):
        if name not in self.gid:
            self.gid[name] = len(self.groups)
            self.groups.append({"name": name, "role": role})
        return self.gid[name]

    # ---- populations ----
    def pop(self, name, region, count, sign=1, group=None, role="input",
            centre=None, radii=None):
        count = max(1, int(round(count)))
        start = self.n
        self.pops[name] = (start, count, sign)
        self.n += count
        rid = RID[region]
        g = self.group_id(group, role) if group else 0
        self.region.append(np.full(count, rid, np.int32))
        self.popid.append(np.full(count, len(self.pop_meta), np.int32))
        self.pop_meta.append({"name": name, "region": rid, "count": count, "sign": sign})
        self.group.append(np.full(count, g, np.int32))
        c = np.array(centre if centre is not None else REGIONS[rid][2], np.float32)
        r = np.array(radii if radii is not None else REGIONS[rid][3], np.float32)
        u = self.rng.normal(size=(count, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        rad = self.rng.random(count) ** (1 / 3)
        self.pos.append((c + u * rad[:, None] * r).astype(np.float32))
        return name

    def idx(self, name):
        s, c, _ = self.pops[name]
        return np.arange(s, s + c, dtype=np.int64)

    # ---- projections: every post cell draws k random presynaptic partners ----
    def proj(self, pre, post, k, drive):
        """drive ~1 means k pre cells firing ~50 Hz push post cells over threshold."""
        pre_names = pre if isinstance(pre, (list, tuple)) else [pre]
        pre_idx = np.concatenate([self.idx(p) for p in pre_names])
        sign = self.pops[pre_names[0]][2]
        post_idx = self.idx(post)
        k = max(1, int(k))
        w = sign * drive * 40.0 / k
        pres = pre_idx[self.rng.integers(0, len(pre_idx), size=(len(post_idx), k))]
        posts = np.repeat(post_idx, k)
        self.edges_pre.append(pres.ravel())
        self.edges_post.append(posts)
        self.edges_w.append(np.full(posts.shape, w, np.float32))

    def arrays(self):
        pre = np.concatenate(self.edges_pre)
        post = np.concatenate(self.edges_post)
        w = np.concatenate(self.edges_w)
        return (np.concatenate(self.region), np.concatenate(self.popid), np.concatenate(self.group),
                np.concatenate(self.pos), pre, post, w)


def build_synthetic(scale, seed):
    b = Builder(seed)
    S = lambda n: n * scale  # noqa: E731
    sides = [("L", 1), ("R", -1)]

    # ---------------- vision ----------------
    for s, sx in sides:
        ol = f"Optic lobe {s}"
        for k in range(SECTORS):
            # sector k spans azimuth front(0) -> back(5); lay out along z
            zc = 90 - k * 36
            b.pop(f"PR_{s}{k}", ol, S(800), group=f"eye_{s}_tonic_{k}",
                  centre=(sx * 320, 0, zc), radii=(12, 110, 18))
            b.pop(f"PRT_{s}{k}", ol, S(200), group=f"eye_{s}_transient_{k}",
                  centre=(sx * 305, 0, zc), radii=(10, 110, 18))
            b.pop(f"MED_{s}{k}", ol, S(3000), centre=(sx * 240, 0, zc * 0.8), radii=(40, 120, 16))
            # OFF cells: tonically active, silenced by light -> respond to dark objects
            b.pop(f"OFF_{s}{k}", ol, S(400), group="bg_off",
                  role="bg", centre=(sx * 230, 0, zc * 0.8), radii=(35, 110, 15))
        b.pop(f"OLI_{s}", ol, S(4000), sign=-1)
        b.pop(f"LPLC2_{s}", ol, 80, centre=(sx * 170, 20, -20), radii=(25, 40, 30))
        b.pop(f"LCoff_{s}", ol, S(300), centre=(sx * 165, 0, 10), radii=(25, 60, 40))
        b.pop(f"LC_{s}", ol, S(2000), centre=(sx * 170, 0, 0), radii=(30, 90, 60))
        b.pop(f"OLfill_{s}", ol, S(10500), group="bg_spont", role="bg")

    # ---------------- olfaction / mushroom body / lateral horn ----------------
    for s, sx in sides:
        al, mb, lh = f"Antennal lobe {s}", f"Mushroom body {s}", f"Lateral horn {s}"
        ant = (sx * 55, -60, 165)
        b.pop(f"ORNa_{s}", al, S(400), group=f"orn_{s}_attractive", centre=ant, radii=(20, 15, 25))
        b.pop(f"ORNv_{s}", al, S(200), group=f"orn_{s}_aversive", centre=ant, radii=(20, 15, 25))
        b.pop(f"ORNo_{s}", al, S(700), group="bg_spont", role="bg", centre=ant, radii=(20, 15, 25))
        b.pop(f"PNa_{s}", al, 60)
        b.pop(f"PNv_{s}", al, 30)
        b.pop(f"PNo_{s}", al, 80)
        b.pop(f"LN_{s}", al, 100, sign=-1)
        b.pop(f"KC_{s}", mb, S(2500))
        b.pop(f"APL_{s}", mb, 1, sign=-1)
        b.pop(f"MBONapp_{s}", mb, 20)
        b.pop(f"MBONavo_{s}", mb, 20)
        b.pop(f"LHNa_{s}", lh, S(150))
        b.pop(f"LHNv_{s}", lh, S(100))
        b.pop(f"LHIa_{s}", lh, 30, sign=-1)
        b.pop(f"LHfill_{s}", lh, S(1150), group="bg_spont", role="bg")

    # ---------------- mechanosensation (Johnston's organ) ----------------
    for s, sx in sides:
        am = f"AMMC / JO {s}"
        b.pop(f"JO_{s}", am, S(480), group=f"jo_{s}", centre=(sx * 60, -40, 150), radii=(15, 15, 20))
        b.pop(f"AMMC_{s}", am, S(300))

    # ---------------- gustation / SEZ ----------------
    gr = (0, -170, 110)
    for s, sx in sides:
        b.pop(f"GRNs_{s}", "SEZ / gustatory", 60, group=f"grn_{s}_sweet",
              centre=(sx * 20, gr[1], gr[2]), radii=(15, 15, 15))
        b.pop(f"GRNb_{s}", "SEZ / gustatory", 40, group=f"grn_{s}_bitter",
              centre=(sx * 20, gr[1], gr[2]), radii=(15, 15, 15))
    b.pop("FEED", "SEZ / gustatory", 200)
    b.pop("BITINH", "SEZ / gustatory", 60, sign=-1)
    b.pop("STOP", "SEZ / gustatory", 40, sign=-1)
    b.pop("MN9", "SEZ / gustatory", 10, group="mn9_proboscis", role="output",
          centre=(0, -130, 60), radii=(10, 10, 10))
    b.pop("SEZfill", "SEZ / gustatory", S(4000), group="bg_spont", role="bg")

    # ---------------- central complex (heading) ----------------
    for k in range(COMPASS):
        a = 2 * np.pi * k / COMPASS
        c = (np.cos(a) * 45, 35 + np.sin(a) * 12, -5)
        b.pop(f"CMP_{k}", "Central complex", 10, group=f"compass_{k}", centre=c, radii=(6, 6, 6))
        b.pop(f"EPG_{k}", "Central complex", 6, group=f"epg_{k}", role="output", centre=c, radii=(6, 6, 6))
    b.pop("D7", "Central complex", 40, sign=-1)
    b.pop("CXfill", "Central complex", S(2900), group="bg_spont", role="bg")

    # ---------------- rest of central brain + descending ----------------
    b.pop("EXPLORE", "Central brain", 50, group="drive_explore", centre=(0, 10, -30), radii=(40, 20, 20))
    for s, sx in sides:
        b.pop(f"WANDER_{s}", "Central brain", 30, group="drive_explore", centre=(sx * 40, 10, -30), radii=(20, 20, 20))
    b.pop("CBe", "Central brain", S(24000), group="bg_spont", role="bg")
    b.pop("CBi", "Central brain", S(7059), sign=-1, group="bg_spont", role="bg")
    for s, sx in sides:
        b.pop(f"DNa02_{s}", "Descending", 5, group=f"dn_turn_{s}", role="output",
              centre=(sx * 12, -130, -45), radii=(4, 20, 4))
        b.pop(f"P9_{s}", "Descending", 5, group=f"dn_forward_{s}", role="output",
              centre=(sx * 8, -140, -50), radii=(4, 20, 4))
    b.pop("MDN", "Descending", 4, group="dn_backward", role="output", centre=(0, -150, -45), radii=(6, 20, 4))
    b.pop("GF", "Descending", 2, group="dn_giant_fiber", role="output", centre=(0, -120, -55), radii=(10, 10, 4))
    b.pop("DNfill", "Descending", S(1280), group="bg_spont", role="bg")

    # ======================= wiring =======================
    for s, o in (("L", "R"), ("R", "L")):
        # vision: photoreceptor -> medulla; light silences OFF cells (sector-local)
        for k in range(SECTORS):
            b.proj(f"PR_{s}{k}", f"MED_{s}{k}", 8, 1.2)
            b.proj(f"MED_{s}{k}", f"OLI_{s}", 1, 0.02)
            b.proj(f"OLI_{s}", f"MED_{s}{k}", 4, 0.4)
            b.proj(f"MED_{s}{k}", f"OFF_{s}{k}", 10, -2.5)  # sign flipped below
            b.proj(f"PRT_{s}{k}", f"LPLC2_{s}", 4, 0.9 / SECTORS)
            b.proj(f"OFF_{s}{k}", f"LCoff_{s}", 6, 2.0 / SECTORS * (1.6 if k < 3 else 0.6))
        b.proj([f"MED_{s}{k}" for k in range(SECTORS)], f"LC_{s}", 12, 0.8)
        b.proj([f"MED_{s}{k}" for k in range(SECTORS)] + [f"LC_{s}"], f"OLfill_{s}", 20, 0.35)
        b.proj(f"OLfill_{s}", f"OLfill_{s}", 15, 0.15)
        # dark object on this side -> turn toward it (Buridan-style fixation)
        b.proj(f"LCoff_{s}", f"DNa02_{s}", 30, 0.55)
        # looming -> giant fibre (both sides converge)
        b.proj(f"LPLC2_{s}", "GF", 40, 0.9)

        # olfaction: ORN -> PN (glomerular), LN gain control
        b.proj(f"ORNa_{s}", f"PNa_{s}", 25, 0.9)
        b.proj(f"ORNv_{s}", f"PNv_{s}", 25, 0.9)
        b.proj(f"ORNo_{s}", f"PNo_{s}", 25, 1.0)
        b.proj([f"ORNa_{s}", f"ORNv_{s}", f"ORNo_{s}"], f"LN_{s}", 30, 0.8)
        for p in ("PNa", "PNv", "PNo"):
            b.proj(f"LN_{s}", f"{p}_{s}", 10, 0.5)
        # PN -> Kenyon cells (~7 claws each), APL feedback inhibition
        b.proj([f"PNa_{s}", f"PNv_{s}", f"PNo_{s}"], f"KC_{s}", 7, 1.0)
        b.proj(f"KC_{s}", f"APL_{s}", 200, 2.0)
        b.proj(f"APL_{s}", f"KC_{s}", 1, 0.4)
        b.proj(f"KC_{s}", f"MBONapp_{s}", 120, 1.5)
        b.proj(f"KC_{s}", f"MBONavo_{s}", 120, 1.5)
        # lateral horn: innate valence
        b.proj(f"PNa_{s}", f"LHNa_{s}", 10, 1.8)
        b.proj(f"PNv_{s}", f"LHNv_{s}", 10, 1.8)
        b.proj([f"PNo_{s}", f"LHNa_{s}", f"LHNv_{s}"], f"LHfill_{s}", 15, 0.5)
        # bilateral contrast: each side's odour inhibits the other side's LHNs
        b.proj(f"PNa_{s}", f"LHIa_{s}", 10, 1.0)
        b.proj(f"LHIa_{s}", f"LHNa_{o}", 10, 0.35)
        # attractive odour: steer toward the stronger side, walk faster
        b.proj(f"LHNa_{s}", f"DNa02_{s}", 30, 1.3)
        # near food flies slow down and search locally (odour does not speed walking)
        b.proj(f"LHNa_{s}", "STOP", 30, 0.12)
        # aversive odour: steer away and back up
        b.proj(f"LHNv_{s}", f"DNa02_{o}", 30, 1.0)
        b.proj(f"LHNv_{s}", "MDN", 20, 0.35)

        # mechanosensation: wind on this antenna -> turn into it
        b.proj(f"JO_{s}", f"AMMC_{s}", 20, 1.0)
        b.proj(f"AMMC_{s}", f"DNa02_{s}", 20, 0.35)
        b.proj(f"AMMC_{s}", "GF", 20, 0.05)

        # gustation
        b.proj(f"GRNs_{s}", "FEED", 20, 0.8)
        b.proj(f"GRNb_{s}", "BITINH", 20, 1.2)
        b.proj(f"GRNb_{s}", "MDN", 20, 0.8)

        # central brain integration & spontaneous chatter
        b.proj([f"LC_{s}", f"MBONapp_{s}", f"MBONavo_{s}", f"LHfill_{s}", f"AMMC_{s}"], "CBe", 4, 0.1)
        # exploration drive -> forward walking
        b.proj("EXPLORE", f"P9_{s}", 25, 0.75)
        # independent noisy drive per side -> spontaneous meandering turns
        b.proj(f"WANDER_{s}", f"DNa02_{s}", 12, 0.3)
        b.proj("STOP", f"P9_{s}", 10, 1.5)
        b.proj("STOP", f"DNa02_{s}", 10, 1.0)

    b.proj("FEED", "MN9", 40, 1.2)
    b.proj("FEED", "STOP", 30, 1.2)
    b.proj("BITINH", "FEED", 20, 2.5)
    b.proj("BITINH", "MN9", 20, 2.5)
    b.proj(["FEED", "BITINH", "MN9"], "SEZfill", 10, 0.4)
    b.proj("SEZfill", "SEZfill", 10, 0.15)

    # central complex ring: compass (sun position) -> EPG wedge, D7 global inhibition
    for k in range(COMPASS):
        b.proj(f"CMP_{k}", f"EPG_{k}", 10, 1.1)
        b.proj(f"EPG_{(k + 1) % COMPASS}", f"EPG_{k}", 3, 0.2)
        b.proj(f"EPG_{(k - 1) % COMPASS}", f"EPG_{k}", 3, 0.2)
        b.proj("D7", f"EPG_{k}", 20, 0.6)
    b.proj([f"EPG_{k}" for k in range(COMPASS)], "D7", 24, 0.7)
    b.proj([f"EPG_{k}" for k in range(COMPASS)] + ["D7"], "CXfill", 10, 0.4)
    b.proj("CXfill", "CXfill", 10, 0.15)

    b.proj("CBe", "CBe", 20, 0.12)
    b.proj("CBi", "CBe", 10, 0.1)
    b.proj("CBe", "CBi", 20, 0.15)
    b.proj(["CBe", "CXfill", "SEZfill"], "DNfill", 30, 0.35)

    region, popid, group, pos, pre, post, w = b.arrays()

    # OFF cells: medulla (light-driven) drive should *inhibit* them
    off = np.zeros(b.n, bool)
    for s in "LR":
        for k in range(SECTORS):
            off[b.idx(f"OFF_{s}{k}")] = True
    med = np.zeros(b.n, bool)
    for s in "LR":
        for k in range(SECTORS):
            med[b.idx(f"MED_{s}{k}")] = True
    flip = off[post] & med[pre]
    w[flip] = -np.abs(w[flip])

    for g in b.groups:
        g["count"] = int(np.sum(group == b.gid[g["name"]]))
    meta = {
        "source": "synthetic",
        "note": "Hand-designed stand-in with FlyWire-scale neuron count and "
                "literature-based pathways. Not the real synapse-level wiring.",
        "w_poisson_mv": 20.0,
        "populations": b.pop_meta,
    }
    return region, popid, group, pos, pre, post, w, b.groups, meta


# ---------------------------------------------------------------------------
# Real FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024; Shiu et al. 2024)
# ---------------------------------------------------------------------------
ANNOT = "annot_Supplemental_file1_neuron_annotations.tsv"
MN9_ID = 720575940660219265  # from Shiu et al. example.ipynb (annotated as CB0701)
ORN_ATTRACTIVE = ["DM1", "DM2", "DM4", "DP1m", "VA2", "VM2"]   # vinegar / fermenting food
ORN_AVERSIVE = ["V", "DA2"]                                     # CO2, geosmin


def build_flywire(raw, seed):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("flywire mode needs pandas + pyarrow (use .venv\\Scripts\\python)")
    rng = np.random.default_rng(seed)
    comp = pd.read_csv(os.path.join(raw, "Completeness_783.csv"), index_col=0)
    ids = comp.index.to_numpy(np.int64)
    n = len(ids)
    con = pd.read_parquet(os.path.join(raw, "Connectivity_783.parquet"),
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    # Bilateral completion of the under-reconstructed right labellar GRNs (tools/mirror_grn.py):
    # synapses are ADDED to existing pairs or as new pairs; disable with FLY_NO_MIRROR=1
    patch = os.path.join(os.path.dirname(raw), "grn_mirror_patch.parquet")
    if os.path.exists(patch) and os.environ.get("FLY_NO_MIRROR") != "1":
        pt = pd.read_parquet(patch, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        con = pd.concat([con, pt]).groupby(["Presynaptic_Index", "Postsynaptic_Index"], as_index=False).sum()
        print(f"  GRN mirror patch applied: {len(pt)} edges, {int(pt['Excitatory x Connectivity'].abs().sum())} synapses added")
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int64)
    w = (0.275 * con["Excitatory x Connectivity"].to_numpy(np.float64)).astype(np.float32)
    del con
    print(f"FlyWire v783: {n} neurons, {len(pre)} connections, {int(np.abs(w).sum() / 0.275 + 0.5)} synapses")

    a = pd.read_csv(os.path.join(raw, ANNOT), sep="\t", low_memory=False).drop_duplicates("root_id")
    a = a.set_index("root_id").reindex(ids)
    print(f"  annotations matched for {a['super_class'].notna().sum()} / {n} neurons")

    def txt(c):
        return a[c].fillna("").astype(str).to_numpy()
    sc, cc, sub, ct, side = txt("super_class"), txt("cell_class"), txt("cell_sub_class"), txt("cell_type"), txt("side")
    # Feeding motor neurons typed by Tastekin et al. 2026 (MN1-MN13, CEM, MNx; muscle targets in
    # data/tastekin2026_flywire_types.tsv) replace the morphological CBxxxx names
    tk_path = os.path.join(os.path.dirname(raw), "tastekin2026_flywire_types.tsv")
    MN_TYPES = []
    if os.path.exists(tk_path):
        tk = pd.read_csv(tk_path, sep="	", comment="#")
        tk = tk[tk["class"] == "motor"]
        pos_of = pd.Series(np.arange(n), index=ids)
        ok = tk.root_id.isin(pos_of.index)
        ct = ct.copy()
        ct[pos_of[tk.root_id[ok]].values] = tk.type[ok].values
        MN_TYPES = sorted(tk.type.unique())
    S = np.where(side == "right", "R", "L")

    # Transmitter signs by Dale's principle (FLY_SIGNS=dale, default; FLY_SIGNS=shiu for the original
    # per-connection signs): each neuron's transmitter is its experimentally known one (known_nt) or
    # else FlyWire's neuron-level prediction (top_nt; Eckstein et al. 2024 Cell). ACh +, GABA/Glu -,
    # dopamine/serotonin/octopamine 0 for FAST transmission (they act through metabotropic receptors;
    # their slow modulatory effects are not modelled). basis: literature
    sign_mode = os.environ.get("FLY_SIGNS", "dale")
    # FLY_GLU_SIGN (diagnostic): glutamate's fast effect. -1 = inhibitory via GluCl (default, as Shiu et al.);
    # 0 = none; +1 = excitatory (fly central synapses also express NMDA/kainate-type receptors)
    GLU_SIGN = float(os.environ.get("FLY_GLU_SIGN", "-1"))
    if sign_mode == "dale":
        knt = a["known_nt"].fillna("").astype(str).str.lower().to_numpy() if "known_nt" in a.columns else np.full(n, "")
        tnt = a["top_nt"].fillna("").astype(str).str.lower().to_numpy() if "top_nt" in a.columns else np.full(n, "")
        nt = np.where(knt != "", knt, tnt)
        def nsign(x):
            if "acetylcholine" in x and "negative" not in x: return 1.0
            if "glutamate" in x: return GLU_SIGN
            if "gaba" in x: return -1.0
            if any(k in x for k in ("dopamine", "serotonin", "octopamine", "tyramine")): return 0.0
            return 1.0 if x == "" else 0.0
        nsg = np.array([nsign(x) for x in nt], np.float32)
        counts = np.abs(w) / 0.275
        w = (0.275 * counts * nsg[pre]).astype(np.float32)
        print(f"  Dale signs: {int((nsg > 0).sum())} excitatory, {int((nsg < 0).sum())} inhibitory, "
              f"{int((nsg == 0).sum())} modulatory (no fast effect) neurons")
    # Input-resistance scaling for very large neurons. A synapse's voltage effect scales with the postsynaptic
    # cell's input resistance, R_in = R_m / A for a passive membrane (gamma = 1, area ~ input synapse count).
    # Shiu et al.'s uniform 0.275 mV/synapse was validated on circuits whose neurons have up to ~6000 input
    # synapses (MN9 ~5.6-5.9k), so it is kept below FLY_SIZE_NREF (default 6000) and the passive correction
    # (w *= (NREF / N_inputs)^gamma) is applied only above it: the ~0.5% giant integrators, mainly
    # multiglomerular AL local neurons and APL (9-17k inputs), which otherwise saturate on spontaneous ORN
    # input. Calibrated on two benchmarks (AL resting rates, sugar -> MN9); see README "Spontaneous activity".
    # FLY_SIZE_NREF= (empty) and FLY_SIZE_GAMMA=0.38: the FlyVis power law over all neurons; FLY_SIZE_GAMMA=0:
    # Shiu et al. uniform weights. basis: biophysics (passive membrane) + approximate threshold
    os.environ.setdefault("FLY_SIZE_NREF", "6000")
    size_gamma = float(os.environ.get("FLY_SIZE_GAMMA", "1.0"))
    size_mode = os.environ.get("FLY_SIZE_MODE", "inputs")
    if size_gamma > 0:
        if size_mode == "inputs":
            # total input synapses per neuron (the FlyVis-fitted rule: strength ~ inputs^-0.38)
            area = np.bincount(post, weights=np.abs(w) / 0.275, minlength=n).astype(np.float64)
            area[area <= 0] = np.median(area[area > 0])
        else:
            cs = pd.read_csv(os.path.join(raw, "cell_stats.csv.gz")).set_index("root_id").area_nm.reindex(ids)
            area = cs.fillna(cs.median()).to_numpy(np.float64)
        nref = float(os.environ.get("FLY_SIZE_NREF", "0")) or float(np.median(area))
        cap = bool(os.environ.get("FLY_SIZE_NREF", ""))     # cap mode: only neurons larger than nref are scaled
        scale = (nref / area) ** size_gamma
        if cap:
            scale = np.minimum(scale, 1.0)
        w = (w * scale[post]).astype(np.float32)
        print(f"  input-resistance scaling gamma={size_gamma}: synapse weight factor 5-95% {np.percentile(scale, 5):.2f}-{np.percentile(scale, 95):.2f}")
    # Photoreceptors (R1-6, R7, R8) release histamine onto histamine-gated chloride channels
    # (Hardie 1989; Gengs et al. 2002), i.e. they INHIBIT their targets. FlyWire's transmitter
    # predictor has no histamine class, so their edges come out mixed; force them inhibitory.
    # basis: literature
    pr = np.isin(ct, ["R1-6", "R7", "R8"])
    flip = pr[pre]
    w[flip] = -np.abs(w[flip])
    print(f"  photoreceptor edges set inhibitory (histamine): {int(flip.sum())}")

    # ---- positions: soma if known, else the annotation point; voxels (4,4,40 nm) -> um ----
    have_soma = a["soma_x"].notna().to_numpy()
    xyz = np.stack([np.where(have_soma, a[f"soma_{k}"], a[f"pos_{k}"]).astype(float) for k in "xyz"], 1)
    xyz *= np.array([0.004, 0.004, 0.040])
    ok = ~np.isnan(xyz).any(1)
    med = np.median(xyz[ok], 0)
    pos = np.zeros((n, 3), np.float32)
    # FlyWire: +x -> fly's right, +y -> ventral, +z -> posterior. Ours: +x left, +y dorsal, +z anterior.
    pos[ok] = -(xyz[ok] - med)

    # ---- regions ----
    region = np.full(n, RID["Central brain"], np.int32)

    def put(mask, name):
        region[mask] = RID[name]
    for sd in "LR":
        m = S == sd
        put(m & np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"]), f"Optic lobe {sd}")
        put(m & (sc == "sensory") & (cc == "visual"), f"Optic lobe {sd}")
        put(m & (((sc == "sensory") & (cc == "olfactory")) | np.isin(cc, ["ALPN", "ALLN", "ALIN", "ALON", "mAL"])), f"Antennal lobe {sd}")
        put(m & np.isin(cc, ["Kenyon_Cell", "MBON", "DAN", "MBIN"]), f"Mushroom body {sd}")
        put(m & np.isin(cc, ["LHLN", "LHCENT"]), f"Lateral horn {sd}")
        put(m & (cc == "mechanosensory") & np.isin(sub, ["auditory", "wind_gravity"]), f"AMMC / JO {sd}")
    put(cc == "CX", "Central complex")
    put((sc == "sensory") & np.isin(cc, ["gustatory", "mechanosensory"]) & ~np.isin(sub, ["auditory", "wind_gravity"]), "SEZ / gustatory")
    put(np.isin(sc, ["motor", "sensory_ascending"]), "SEZ / gustatory")
    put(sc == "descending", "Descending")
    put(sc == "ascending", "Ascending")
    for i in np.where(~ok)[0]:  # no coordinate: sample inside the region ellipsoid
        _, _, cen, rad = REGIONS[region[i]]
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        pos[i] = np.array(cen) + u * rng.random() ** (1 / 3) * np.array(rad)

    # ---- populations: annotated class x hemisphere (plus key named cell types) ----
    cls = np.where(cc != "", cc, np.where(sc != "", sc, "unannotated"))
    key = np.array([f"{s}:{c}_{d}" for s, c, d in zip(sc, cls, S)], dtype=object)
    for t in ["DNa02", "DNp09", "MDN", "DNp01", "LPLC2", "EPG", "APL", "CB0701", "R7", "R8", "L1", "L2", "L3",
              "R1-6", "L4", "L5", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4a", "T4b", "T4c", "T4d",
              "T5a", "T5b", "T5c", "T5d", "LC4", "LPLC1", "LC6", "DNp02", "DNp11"] + MN_TYPES:
        m = ct == t
        key[m] = np.array([f"type:{t}_{d}" for d in S[m]], dtype=object)
    uniq, popid = np.unique(key.astype(str), return_inverse=True)
    popid = popid.astype(np.int32)
    pop_meta = []
    for i, k in enumerate(uniq):
        m = popid == i
        pop_meta.append({"name": str(k), "region": int(np.bincount(region[m]).argmax()), "count": int(m.sum()), "sign": 1})

    # ---- sensory / motor groups ----
    groups = [{"name": "none", "role": "none"}]
    gid = {"none": 0}
    group = np.zeros(n, np.int32)

    def assign(mask, name, role="input"):
        if name not in gid:
            gid[name] = len(groups)
            groups.append({"name": name, "role": role})
        group[mask & (group == 0)] = gid[name]

    def sectors(mask, inverted):
        """Split cells into SECTORS front->back bins along the anterior axis.
        Lamina is retinotopically upright; medulla (R7/R8 terminals) is inverted by the chiasm."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        z = pos[idx, 2] * (-1 if inverted else 1)
        q = np.clip((np.argsort(np.argsort(-z)) * SECTORS) // len(idx), 0, SECTORS - 1)
        out = []
        for k in range(SECTORS):
            mm = np.zeros(n, bool)
            mm[idx[q == k]] = True
            out.append((mm, k))
        return out

    for sd in "LR":
        m = S == sd
        for mm, k in sectors(m & np.isin(ct, ["R7", "R8"]), inverted=True):
            assign(mm, f"eye_{sd}_tonic_{k}")
        for t, g in (("L3", "tonic"), ("L2", "transient"), ("L1", "on")):
            for mm, k in sectors(m & (ct == t), inverted=False):
                assign(mm, f"eye_{sd}_{g}_{k}")
        assign(m & (cc == "visual") & (sub == "DRA"), f"dra_{sd}")
        assign(m & (cc == "visual") & (sub == "ocellar"), "ocelli")
        orn = (sc == "sensory") & (cc == "olfactory")
        assign(m & orn & np.isin(ct, ["ORN_" + g for g in ORN_ATTRACTIVE]), f"orn_{sd}_attractive")
        assign(m & orn & np.isin(ct, ["ORN_" + g for g in ORN_AVERSIVE]), f"orn_{sd}_aversive")
        assign(m & (cc == "mechanosensory") & (sub == "wind_gravity"), f"jo_{sd}")
        assign(m & (cc == "mechanosensory") & (sub == "auditory"), f"jo_auditory_{sd}")
        gus = cc == "gustatory"
        assign(m & gus & np.isin(sub, ["sugar", "sugar/low_salt"]), f"grn_{sd}_sweet")
        assign(m & gus & (sub == "bitter"), f"grn_{sd}_bitter")
        assign(m & gus & (sub == "water"), f"grn_{sd}_water")
        assign(m & (ct == "DNa02"), f"dn_turn_{sd}", "output")
        assign(m & (ct == "DNp09"), f"dn_forward_{sd}", "output")
    assign((cc == "thermosensory") & (sub == "heating"), "thermo_hot")
    assign((cc == "thermosensory") & (sub == "cold"), "thermo_cold")
    assign((cc == "hygrosensory") & (sub == "moist"), "hygro_moist")
    assign((cc == "hygrosensory") & (sub == "dry"), "hygro_dry")
    assign(ct == "MDN", "dn_backward", "output")
    assign(ct == "DNp01", "dn_giant_fiber", "output")
    assign((ids == MN9_ID) | (ct == "CB0701") | (ct == "MN9"), "mn9_proboscis", "output")
    for t in MN_TYPES:
        if t != "MN9":
            assign(ct == t, "mn_" + t, "output")
    assign(ct == "EPG", "epg_all", "output")
    gap_junctions = []
    # ---- ventral nerve cord from BANC (tools/build_vnc.py): new neurons + VNC synapses
    vnc_dir = os.path.join(os.path.dirname(raw), "vnc")
    if os.path.exists(os.path.join(vnc_dir, "vnc_neurons.csv")) and os.environ.get("FLY_NO_VNC") != "1":
        vn = pd.read_csv(os.path.join(vnc_dir, "vnc_neurons.csv"))
        ve = pd.read_parquet(os.path.join(vnc_dir, "vnc_edges.parquet"))
        nv = len(vn)
        assert (vn.model_index.values == n + np.arange(nv)).all()
        vsc = vn.super_class.fillna("").values
        vreg = np.where(vsc == "motor", RID["VNC motor"], np.where(np.isin(vsc, ["sensory", "sensory_ascending", "sensory_descending"]),
                        RID["VNC sensory"], RID["VNC interneurons"])).astype(np.int32)
        vpos = np.zeros((nv, 3), np.float32)
        for k in range(nv):
            _, _, cen, rad = REGIONS[vreg[k]]
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            vpos[k] = np.array(cen) + u * rng.random() ** (1 / 3) * np.array(rad)
        vside = np.where(vn.side.astype(str) == "right", "R", "L")
        VNC_TYPES = ["TTMn", "PSI", "DLM1-4", "DLM5", "tergotrochanter"]
        vkey = np.array([f"vnc_type:{t}_{sd}" if t in VNC_TYPES else
                         (f"vnc:{c}_{bp}_{sd}" if c == "leg_motor_neuron" else f"vnc:{c}_{sd}")
                         for t, c, bp, sd in zip(vn.cell_type.astype(str), vn.cell_class.fillna("unknown").astype(str),
                                                 vn.body_part.fillna("").astype(str), vside)], dtype=object)
        allkey = np.concatenate([uniq[popid].astype(object), vkey])
        uniq, popid = np.unique(allkey.astype(str), return_inverse=True)
        popid = popid.astype(np.int32)
        region = np.concatenate([region, vreg])
        pos = np.concatenate([pos, vpos])
        group = np.concatenate([group, np.zeros(nv, np.int32)])
        ids = np.concatenate([ids, vn.bid.to_numpy(np.int64)])
        ve_w = (0.275 * ve["Excitatory x Connectivity"].to_numpy(np.float64)).astype(np.float32)
        if GLU_SIGN != -1.0:
            # re-sign glutamatergic presynaptic neurons: resident (BANC code) or bridged FlyWire (Dale nt)
            pre_v = ve.Presynaptic_Index.to_numpy()
            glu_res = (vn.nt.astype(str).str.upper().values == "GLUT")
            is_glu = np.where(pre_v >= n, glu_res[np.clip(pre_v - n, 0, nv - 1)],
                              np.array(["glutamate" in x for x in nt])[np.clip(pre_v, 0, n - 1)])
            ve_w = np.where(is_glu, np.abs(ve_w) * GLU_SIGN, ve_w).astype(np.float32)
        if size_gamma > 0:
            if size_mode == "inputs":
                vin = np.bincount(ve.Postsynaptic_Index.to_numpy() - 0, weights=ve.Connectivity.to_numpy(), minlength=len(area) + nv)
                area_all = np.concatenate([area, vin[len(area):]]).astype(np.float64)
                area_all[:len(area)] += vin[:len(area)]          # bridged FlyWire DN/AN: add their VNC inputs
                area_all[area_all <= 0] = np.median(area)
            else:
                bnn = pd.read_csv(os.path.join(raw, "banc", "neurons.csv.gz")).set_index("Root ID")["Surface area (nm^2)"]
                area_all = np.concatenate([area, bnn.reindex(vn.bid).fillna(np.median(area)).to_numpy(np.float64)])
            vs = (nref / area_all[ve.Postsynaptic_Index.to_numpy()]) ** size_gamma
            ve_w = (ve_w * (np.minimum(vs, 1.0) if cap else vs)).astype(np.float32)
        pre = np.concatenate([pre, ve.Presynaptic_Index.to_numpy(np.int64)])
        post = np.concatenate([post, ve.Postsynaptic_Index.to_numpy(np.int64)])
        w = np.concatenate([w, ve_w])
        # documented gap junctions (giant-fibre system), data/vnc/electrical_synapses.json: stored as a separate
        # list (direct voltage coupling in lif.glsl), not as chemical synapses
        el = json.load(open(os.path.join(vnc_dir, "electrical_synapses.json")))
        for pr in el["pairs"]:
            for sd, sdn in (("R", "right"), ("L", "left")):
                pres = np.where((ct == pr["pre_type"]) & (S == sd))[0]
                posts = n + np.where((vn.cell_type.astype(str).values == pr["post_type"]) & (vn.side.astype(str).values == sdn))[0]
                for a_ in pres:
                    for b_ in posts:
                        gap_junctions.append([int(a_), int(b_), float(pr["coupling_mv"])])
        n_el = len(gap_junctions)
        n += nv
        for t, g_ in (("TTMn", "vnc_jump_ttmn"), ("DLM1-4", "vnc_flight_dlm"), ("PSI", "vnc_psi")):
            m_ = np.zeros(n, bool); m_[len(group) - nv:] = (vn.cell_type.astype(str).values == t)
            if g_ not in gid:
                gid[g_] = len(groups); groups.append({"name": g_, "role": "output"})
            group[m_ & (group == 0)] = gid[g_]
        print(f"  VNC attached: {nv} neurons, {len(ve)} VNC edges, {n_el} gap junctions -> {n} neurons total")
        pop_meta = []
        for i, k in enumerate(uniq):
            m_ = popid == i
            pop_meta.append({"name": str(k), "region": int(np.bincount(region[m_]).argmax()), "count": int(m_.sum()), "sign": 1})
    # Peripheral afferents (ORNs, GRNs, mechanosensory, photoreceptors...) initiate spikes in the periphery
    # (antenna, labellum, legs); FlyWire/BANC synapses ONTO their axon terminals (ORN-ORN, LN->ORN,
    # GABAergic feedback onto GRNs) act presynaptically on transmitter release and cannot fire the cell
    # (Olsen & Wilson 2008 Nature; Root et al. 2008 Cell; Horne et al. 2018 eLife). In a point-neuron LIF
    # they would wrongly drive spikes, so they are removed (FLY_AFFERENT_INPUTS=1 keeps them, Shiu et al.).
    # Presynaptic gain control itself is not modelled. basis: literature (approximation: omission)
    if os.environ.get("FLY_AFFERENT_INPUTS", "0") != "1":
        aff = np.zeros(n, bool)
        aff[:len(sc)] = sc == "sensory"
        if len(sc) < n:
            aff[len(sc):] = np.isin(vsc, ["sensory", "sensory_ascending", "sensory_descending"])
        cut = aff[post]
        print(f"  afferent axon-terminal inputs removed: {int(cut.sum())} edges onto {int(aff.sum())} sensory neurons")
        pre, post, w = pre[~cut], post[~cut], w[~cut]
    for g in groups:
        g["count"] = int(np.sum(group == gid[g["name"]]))
    print(f"  {len(uniq)} populations, {len(groups) - 1} sensory/motor groups:")
    print("   " + ", ".join(f"{g['name']}={g['count']}" for g in groups[1:]))
    meta = {
        "source": "flywire_v783",
        "note": "Real FlyWire v783 connectome (Dorkenwald et al. 2024), annotations (Schlegel et al. 2024), "
                "LIF weights and parameters from Shiu et al. 2024. Sector mapping of eye inputs is approximate.",
        "w_poisson_mv": 68.75,  # Shiu et al.: f_poi = 250 x w_syn
        "populations": pop_meta,
        "lif_overrides": {"delay_ms": 1.8, "reset_g_on_spike": True},
        "gap_junctions": gap_junctions,
        # body calibration for single identified descending neurons (edit freely)
        "motor": {"fwd_offset_hz": 0.0, "fwd_full_hz": 40.0, "back_full_hz": 30.0,
                  "turn_full_hz": 45.0, "mn9_on_hz": 10.0},
    }
    return region, popid, group, pos, pre, post, w, groups, meta


# ---------------------------------------------------------------------------
def write(region, popid, group, pos, pre, post, w, groups, meta, name="connectome"):
    n = len(region)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    row_ptr = np.zeros(n + 1, np.int32)
    np.cumsum(np.bincount(pre, minlength=n), out=row_ptr[1:])
    wq = np.clip(np.round(w * FIXED), -2**31 + 1, 2**31 - 1).astype(np.int32)
    e = len(post)

    # viewer: 16 floats / instance (3x4 transform rows + colour), scaled to Godot units
    n_pad = ((n + TEX_W - 1) // TEX_W) * TEX_W
    scale = 1.0 / 100.0
    mm = np.zeros((n, 16), np.float32)
    mm[:, 0] = mm[:, 5] = mm[:, 10] = 1.0
    mm[:, 3] = pos[:, 0] * scale
    mm[:, 7] = pos[:, 1] * scale
    mm[:, 11] = pos[:, 2] * scale
    cols = np.array([[int(REGIONS[r][1][i:i + 2], 16) / 255 for i in (1, 3, 5)] for r in range(len(REGIONS))], np.float32)
    mm[:, 12:15] = cols[region]
    mm[:, 15] = 1.0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name + ".bin"), "wb") as f:
        f.write(b"FLYC")
        f.write(struct.pack("<iiiii", 2, n, e, n_pad, len(groups)))
        f.write(popid.astype(np.int32).tobytes())
        f.write(group.astype(np.int32).tobytes())
        f.write(row_ptr.tobytes())
        f.write(post.astype(np.int32).tobytes())
        f.write(wq.tobytes())
        f.write(mm.tobytes())

    region_counts = np.bincount(region, minlength=len(REGIONS))
    meta.update({
        "neurons": n, "synapse_edges": e, "tex_width": TEX_W, "n_pad": n_pad,
        "regions": [{"name": r[0], "color": r[1], "count": int(region_counts[i])} for i, r in enumerate(REGIONS)],
        "groups": groups,
        "lif": {"v_rest": -52.0, "v_reset": -52.0, "v_th": -45.0, "tau_m_ms": 20.0,
                "tau_syn_ms": 5.0, "refractory_ms": 2.2, "dt_ms": 0.5,
                "delay_ms": 0.5, "reset_g_on_spike": False},
    })
    meta["lif"].update(meta.pop("lif_overrides", {}))
    with open(os.path.join(OUT_DIR, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {n} neurons, {e} synaptic edges -> data/{name}.bin "
          f"({os.path.getsize(os.path.join(OUT_DIR, name + '.bin')) / 1e6:.1f} MB)")
    for i, r in enumerate(REGIONS):
        print(f"  {r[0]:<18} {region_counts[i]:>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "flywire"])
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic: population size multiplier")
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "synthetic":
        write(*build_synthetic(a.scale, a.seed), name="connectome_synthetic")
    else:
        write(*build_flywire(a.raw, a.seed), name="connectome_flywire")


if __name__ == "__main__":
    main()
