"""Map the named feeding-circuit neurons of Shiu, Sterne et al. 2022 eLife 11:e79887 (their FlyWire IDs are from an
older release) to FlyWire v783 by soma position: CATMAID FAFB14 soma coordinates (public VFB server,
fafb.catmaid.virtualflybrain.org, skeleton IDs from the paper) -> nearest v783 soma; accepted if <= 2 um. Writes data/named_feeding_neurons.csv.
Validated on neurons with a known v783 identity (Zorro R = CB0192, MN9 = CB0701)."""
import pathlib, numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = pathlib.Path(__file__).resolve().parents[1]
# soma position per skeleton: node tagged "soma" where present, else the node with the largest radius
# (retrieved 2026-10-03 via /1/skeletons/<id>/compact-detail?with_tags=true); "src" records which
CATMAID_SOMA_NM = {   # name: (skeleton id, x, y, z, src)
 "Billiards": (8606542, 512572.47, 357134.5, 168960, "soma"), "Bract1": (17024882, 522635.1, 365352.25, 177440, "soma"),
 "Bract2": (17542353, 526361, 370939, 176040, "radius"), "Clavicle": (10150139, 484336.34, 355220, 120920, "none"),
 "Dandelion": (17249809, 501952, 329856, 110320, "radius"), "Fdg": (16783943, 470253.62, 322942.22, 113720, "soma"),
 "FMIn": (8952676, 503966.22, 367823.28, 165520, "soma"), "Fuchs": (7929209, 483725.44, 366416.97, 132880, "radius"),
 "Fudog": (7983275, 419941.84, 318245, 128200, "soma"), "G2N-1": (15079937, 536864, 363328, 121520, "radius"),
 "MN9": (16866694, 583621, 326227, 43360, "radius"), "Phantom": (16762541, 441790, 332233, 69200, "radius"),
 "Quasimodo": (8275570, 438297, 340241, 82920, "soma"), "Rattle": (16238926, 466112, 328384, 107520, "radius"),
 "Rounddown": (16886973, 420227.25, 316893.2, 146040, "radius"), "Roundup": (16002203, 431380, 338308, 104040, "radius"),
 "Scapula": (16887116, 423811.88, 324645.5, 116320, "radius"), "Specter": (17579359, 545728, 330752, 100800, "radius"),
 "Sternum": (17533840, 378396, 293288, 125840, "radius"), "Usnea": (14890522, 430897, 327474, 78320, "soma"),
 "Zorro_L": (7574284, 621934.56, 346382.88, 110280, "soma"), "Zorro_R": (7899212, 486672.5, 365313.78, 146760, "soma")}
ACCEPT_UM = 2.0       # validation: MN9 -> CB0701 at 0.3 um, Zorro -> CB0192 at 0.6-0.7 um; misses are > 5 um
comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                usecols=["root_id", "soma_x", "soma_y", "soma_z", "cell_type", "side", "super_class", "top_nt"]).drop_duplicates("root_id")
a = a[a.root_id.isin(comp.index) & a.soma_x.notna()].reset_index(drop=True)
xyz = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * np.array([4, 4, 40])
tree = cKDTree(xyz)
rows = []
for n, (sk, x, y, z, src) in CATMAID_SOMA_NM.items():
    d, i = tree.query([x, y, z], k=3)
    r = a.iloc[i[0]]
    rows.append({"name": n, "catmaid_skeleton": sk, "root_id": int(r.root_id), "cell_type": r.cell_type, "side": r.side,
                 "top_nt": r.top_nt, "dist_um": round(d[0] / 1000, 1), "second_um": round(d[1] / 1000, 1),
                 "second_type": a.iloc[i[1]].cell_type, "soma_source": src,
                 "accepted": bool(d[0] / 1000 <= ACCEPT_UM)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/named_feeding_neurons.csv", index=False)
print(df.to_string())
