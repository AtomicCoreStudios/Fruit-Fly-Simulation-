"""Shape- and wiring-based identification of leg sensory neurons that BANC leaves unlabelled or (probably) mislabelled,
in particular the middle/hind-leg campaniform sensilla (load sensors) that BANC annotates only in the front legs.

Two independent descriptors per neuron (BANC v626, Bates et al. 2025):
  morphology : the axon's arbor inside its own leg neuromere, in a common frame: each neuromere's cloud of
               FeCO + bristle axons (labelled in every leg; left legs mirrored) is centred, scaled and rigidly
               registered (ICP) onto the front-right neuromere. Descriptor = Gaussian-smoothed 3D occupancy histogram of the skeleton nodes.
  wiring     : output synapses (and, separately, input synapses) split by partner hemilineage / partner class.
               Hemilineages repeat in every thoracic segment, so the descriptor is comparable across legs.
A third descriptor, wiring by partner CELL TYPE (serial types shared across legs), lets neurons without a
skeleton be classified from two independent wiring views.
Classifier per descriptor: multinomial logistic regression (scikit-learn), trained on labelled neurons:
  front-leg campaniform sensilla and hair plates (the only legs where BANC labels both), FeCO claw/hook/club,
  multidendritic, bristle, taste and tactile-taste bristles (all legs; bristle/taste subsampled).
Validation: stratified 5-fold cross-validation, front-left -> front-right transfer, and front -> middle/hind transfer
for the classes labelled in every leg (tests that the per-neuromere frame generalises across segments).
Re-labelling rule ("proven"): two independent classifiers (morphology + wiring, or, without a skeleton, the two
wiring views) predict the same class, each with probability >= P_MIN; each rule's precision is measured by training on
front legs and testing on labelled middle/hind neurons. Otherwise the neuron keeps its BANC label ("uncertain").
Targets: middle/hind-leg hair plates and campaniform sensilla, orphan leg sensory neurons, untyped leg chordotonal.
usage: .venv/Scripts/python tools/classify_leg_sensory.py -> data/leg_sensory_inferred.csv, recordings/leg_sensory_cv.txt"""
import io, zipfile, pathlib, re
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
P_MIN, NB, SPAN, RNG = 0.8, 8, 2.5, np.random.default_rng(783)
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
v["leg"] = v.sub_class.astype(str).str.extract(r"^(front|middle|hind)_leg")[0]
v["kind"] = v.sub_class.astype(str).str.replace(r"^(front|middle|hind)_leg_", "", regex=True).str.replace("_neuron", "")
LABEL = {"campaniform_sensillum": "campaniform", "bilateral_campaniform_sensillum": "campaniform_bilateral",
         "hair_plate": "hair_plate", "claw_chordotonal_organ": "FeCO_claw", "hook_chordotonal_organ": "FeCO_hook",
         "club_chordotonal_organ": "FeCO_club", "multidendritic": "multidendritic", "bristle": "bristle",
         "taste_bristle_gustatory": "taste_gustatory", "taste_bristle_tactile": "taste_tactile"}
sens = v[(v.super_class == "sensory") & v.leg.notna()].copy()
sens["label"] = sens.kind.map(LABEL)
# training set: CS/HP only from the front legs; everything else labelled in all legs; big classes subsampled
is_train = sens.label.notna() & ~((sens.label.isin(["campaniform", "hair_plate"])) & (sens.leg != "front"))
train = sens[is_train]
parts = []
for lab, g in train.groupby("label"):
    parts.append(g.sample(n=min(len(g), 240), random_state=1) if lab in ("bristle", "taste_gustatory") else g)
train = pd.concat(parts)
target = sens[(sens.label.isin(["campaniform", "hair_plate"]) & (sens.leg != "front")) | sens.kind.isin(["orphan", "chordotonal_organ"])]
use = pd.concat([train, target]).drop_duplicates("model_index")
print("training neurons:", train.label.value_counts().to_dict())
print("targets:", target.groupby(["leg", "kind"]).size().to_dict())

# ---- skeletons ----
z = zipfile.ZipFile(ROOT / "data/raw/banc/neuron_skeletons.zip")
names = {int(pathlib.Path(n).stem): n for n in z.namelist() if n.endswith(".swc")}


def nodes(bid):
    n = names.get(int(bid))
    if n is None:
        return None
    a = np.loadtxt(io.BytesIO(z.read(n)), usecols=(2, 3, 4), ndmin=2)
    return a if len(a) > 3 else None


# per-neuromere frames: the axon arbors of the sensory classes labelled in every leg (FeCO claw/hook/club, bristles)
# form a point cloud per neuromere; each neuromere (left legs mirrored) is centred, scaled and rigidly registered
# (ICP, Kabsch) onto the front-right neuromere, so arbors are compared in one anatomical frame
from scipy.spatial import cKDTree
mn = v[(v.cell_class == "leg_motor_neuron") & v.leg.notna()]
lr = {}
for side in ("left", "right"):
    pts = [nodes(b) for b in mn[mn.side == side].bid.sample(n=60, random_state=0)]
    lr[side] = np.concatenate([p for p in pts if p is not None]).mean(0)
d_lr = lr["right"] - lr["left"]; ml_axis = int(np.argmax(np.abs(d_lr))); ml_sign = np.sign(d_lr[ml_axis])
mid_x = (lr["right"][ml_axis] + lr["left"][ml_axis]) / 2
ref_lab = sens[sens.kind.isin(["claw_chordotonal_organ", "hook_chordotonal_organ", "club_chordotonal_organ", "bristle"])]
clouds = {}
for (leg, side), g in ref_lab.groupby(["leg", "side"]):
    pts = [nodes(b) for b in g.bid.sample(n=min(len(g), 150), random_state=0)]
    P = np.concatenate([p for p in pts if p is not None])
    if (side == "left") == (ml_sign > 0):
        P[:, ml_axis] = 2 * mid_x - P[:, ml_axis]                        # mirror across the midline
    clouds[(leg, side)] = P[RNG.choice(len(P), size=min(len(P), 30000), replace=False)]


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    return R, cb - R @ ca


ref = clouds[("front", "right")]; rc = ref.mean(0); rs = np.sqrt(((ref - rc) ** 2).sum(1).mean())
refn = (ref - rc) / rs; tree = cKDTree(refn)
frame = {}
for k, P in clouds.items():
    c = P.mean(0); sc = np.sqrt(((P - c) ** 2).sum(1).mean()); Q = (P - c) / sc
    R_tot, t_tot = np.eye(3), np.zeros(3)
    for it in range(40):
        Qt = Q @ R_tot.T + t_tot
        dd, ii = tree.query(Qt)
        keep_ = dd <= np.percentile(dd, 90)
        R, t = kabsch(Qt[keep_], refn[ii[keep_]])
        R_tot, t_tot = R @ R_tot, R @ t_tot + t
    frame[k] = (c, sc, R_tot, t_tot, float(np.median(dd)))
print("medio-lateral axis", "xyz"[ml_axis], "| ICP median residual (ref-scale units):", {f"{a}_{b[0]}": round(f[4], 3) for (a, b), f in frame.items()})


def morph_desc(bid, leg, side):
    a = nodes(bid)
    if a is None or (leg, side) not in frame:
        return None
    if (side == "left") == (ml_sign > 0):
        a = a.copy(); a[:, ml_axis] = 2 * mid_x - a[:, ml_axis]
    c, sc, R, t, _ = frame[(leg, side)]
    q = ((a - c) / sc) @ R.T + t
    q = q[np.all(np.abs(q) < SPAN, axis=1)]                                  # arbor inside the neuromere
    if len(q) < 5:
        return None
    h, _ = np.histogramdd(q, bins=NB, range=[(-SPAN, SPAN)] * 3)
    h = gaussian_filter(h, 0.8).ravel()
    return h / max(np.linalg.norm(h), 1e-9)


# ---- wiring ----
e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
vi = v.set_index("model_index")
SC, HL, CT, FN, SUB = (vi.super_class.astype(str).to_dict(), vi.hemilineage.to_dict(), vi.cell_type.astype(str).to_dict(),
                       vi.function.astype(str).to_dict(), vi.sub_class.astype(str).to_dict())


def partner_key(i, level):
    """level 'hl': interneurons by hemilineage; 'ct': interneurons by cell type (serial types shared across legs)."""
    sc = SC.get(i); sc = None if sc is None else str(sc)
    if sc is None:
        return "brain"
    if sc == "ventral_nerve_cord_intrinsic":
        if level == "ct":
            return "ct_" + str(CT.get(i, "?"))
        h = HL.get(i)
        return "hl_" + (h if isinstance(h, str) else "unknown")
    if sc == "motor":
        f = str(FN.get(i, "")).replace("leg_motor", "").strip(",")
        return "mn_" + (f.split(",")[0] if f else "other")
    if sc == "sensory":
        return "sn_" + re.sub(r"^(front|middle|hind)_leg_", "", str(SUB.get(i, "")))
    return "sc_" + str(sc)


ids = set(use.model_index)
eo = e[e.Presynaptic_Index.isin(ids)]; ei = e[e.Postsynaptic_Index.isin(ids)]
Wd = {}
for level in ("hl", "ct"):
    o = eo.assign(k=[partner_key(i, level) for i in eo.Postsynaptic_Index]).pivot_table(
        index="Presynaptic_Index", columns="k", values="Connectivity", aggfunc="sum", fill_value=0)
    n_ = ei.assign(k=[partner_key(i, level) for i in ei.Presynaptic_Index]).pivot_table(
        index="Postsynaptic_Index", columns="k", values="Connectivity", aggfunc="sum", fill_value=0)
    o = o.div(o.sum(1), axis=0).add_prefix("out_"); n_ = n_.div(n_.sum(1).replace(0, 1), axis=0).add_prefix("in_")
    Wd[level] = o.join(n_, how="outer").fillna(0.0)

use = use.set_index("model_index")
use = use[use.index.isin(Wd["hl"].index)]
Mrows = {}
for i, r in use.iterrows():
    d = morph_desc(r.bid, r.leg, r.side)
    if d is not None:
        Mrows[i] = d
has_m = use.index.isin(list(Mrows))
print("neurons with synapses:", len(use), "| also with a skeleton in the neuromere:", int(has_m.sum()),
      "| missing skeleton by group:", use[~has_m].groupby(["leg", "kind"]).size().to_dict())
X = {"morph": np.array([Mrows.get(i, np.zeros(NB ** 3)) for i in use.index]),
     "wiring_hl": Wd["hl"].reindex(use.index).fillna(0).to_numpy(), "wiring_ct": Wd["ct"].reindex(use.index).fillna(0).to_numpy()}
tr = (use.label.notna() & use.index.isin(train.model_index)).to_numpy()
ylab = use.label.to_numpy()
EVERY = ["FeCO_claw", "FeCO_hook", "FeCO_club", "bristle", "taste_gustatory", "taste_tactile", "multidendritic"]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.5))


rep = []
for nm, Xa in X.items():
    sel = tr & (has_m if nm == "morph" else True)
    Xt, yt = Xa[sel], ylab[sel]; sub = use[sel]
    cv = cross_val_predict(model(), Xt, yt, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    labs = sorted(set(yt)); cs_hp = np.isin(yt, ["campaniform", "hair_plate"])
    rep.append(f"{nm}: 5-fold CV accuracy {accuracy_score(yt, cv):.3f}; campaniform vs hair plate {accuracy_score(yt[cs_hp], cv[cs_hp]):.3f} "
               f"(n={cs_hp.sum()}, majority-class baseline {max(np.mean(yt[cs_hp] == 'hair_plate'), np.mean(yt[cs_hp] == 'campaniform')):.3f})")
    rep.append("  confusion (rows true, cols predicted) " + str(labs) + "\n" + str(confusion_matrix(yt, cv, labels=labs)))
    fl = ((sub.leg == "front") & (sub.side == "left")).to_numpy(); fr = ((sub.leg == "front") & (sub.side == "right")).to_numpy()
    rep.append(f"  front-left -> front-right transfer {accuracy_score(yt[fr], model().fit(Xt[fl], yt[fl]).predict(Xt[fr])):.3f}")
    ev = np.isin(yt, EVERY); f_ = (sub.leg == "front").to_numpy() & ev; o_ = (sub.leg != "front").to_numpy() & ev
    rep.append(f"  front -> middle/hind transfer (classes labelled in every leg) {accuracy_score(yt[o_], model().fit(Xt[f_], yt[f_]).predict(Xt[o_])):.3f}")

# acceptance rules and their measured precision: train on FRONT-leg labels only, apply to labelled middle/hind neurons
RULES = {"morph+wiring_hl": ("morph", "wiring_hl"), "wiring_hl+wiring_ct": ("wiring_hl", "wiring_ct")}
sub = use[tr]; ftr = (sub.leg == "front").to_numpy(); oth = (sub.leg != "front").to_numpy() & np.isin(ylab[tr], EVERY)
prec = {}
for rn, (a_, b_) in RULES.items():
    need_m = "morph" in (a_, b_)
    f_ = ftr & (has_m[tr] if need_m else True); o_ = oth & (has_m[tr] if need_m else True)
    ma = model().fit(X[a_][tr][f_], ylab[tr][f_]); mb = model().fit(X[b_][tr][f_], ylab[tr][f_])
    Pa, Pb = ma.predict_proba(X[a_][tr][o_]), mb.predict_proba(X[b_][tr][o_])
    ca, cb = ma.classes_[Pa.argmax(1)], mb.classes_[Pb.argmax(1)]
    ok = (ca == cb) & (Pa.max(1) >= P_MIN) & (Pb.max(1) >= P_MIN)
    prec[rn] = float(np.mean(ca[ok] == ylab[tr][o_][ok])) if ok.any() else float("nan")
    rep.append(f"rule {rn} (agree, both p >= {P_MIN}), trained on front legs, tested on {o_.sum()} labelled middle/hind "
               f"neurons: accepts {ok.sum()} ({ok.mean():.0%}), correct {prec[rn]:.1%}")
print("\n".join(rep))
(ROOT / "recordings/leg_sensory_cv.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")

# ---- final calls on the targets (models trained on all training labels) ----
fit = {nm: model().fit(Xa[tr & (has_m if nm == "morph" else True)], ylab[tr & (has_m if nm == "morph" else True)]) for nm, Xa in X.items()}
tg = ~tr
res = use[tg][["bid", "leg", "side", "kind", "cell_type"]].copy()
for nm in X:
    P = fit[nm].predict_proba(X[nm][tg]); res[f"pred_{nm}"] = fit[nm].classes_[P.argmax(1)]; res[f"p_{nm}"] = P.max(1).round(3)
res["has_skeleton"] = has_m[tg]
call, rule = [], []
for r in res.itertuples():
    if r.has_skeleton and r.pred_morph == r.pred_wiring_hl and r.p_morph >= P_MIN and r.p_wiring_hl >= P_MIN:
        call.append(r.pred_morph); rule.append(f"morph+wiring_hl (precision {prec['morph+wiring_hl']:.0%})")
    elif r.pred_wiring_hl == r.pred_wiring_ct and r.p_wiring_hl >= P_MIN and r.p_wiring_ct >= P_MIN:
        call.append(r.pred_wiring_hl); rule.append(f"wiring_hl+wiring_ct (precision {prec['wiring_hl+wiring_ct']:.0%})")
    else:
        call.append("uncertain"); rule.append("")
res["inferred"] = call; res["rule"] = rule
res["basis"] = "inferred from BANC morphology/wiring vs labelled leg sensory neurons (tools/classify_leg_sensory.py)"
res.reset_index().to_csv(ROOT / "data/leg_sensory_inferred.csv", index=False)
print("\nresults by BANC label and leg:")
print(pd.crosstab([res.kind, res.leg], res.inferred).to_string())

# ---- validated re-labels -> data/leg_sensory_relabel.json (used by tools/build_leg_map.py) ----
# only classes whose acceptance rule was validated on labelled middle/hind neurons (labelled in every leg); campaniform /
# hair-plate calls cannot be validated that way (BANC labels both only in the front legs) and are NOT used
import json
BANC2 = {"claw_chordotonal_organ": "FeCO_claw", "hook_chordotonal_organ": "FeCO_hook", "club_chordotonal_organ": "FeCO_club",
         "hair_plate": "hair_plate", "campaniform_sensillum": "campaniform", "bristle": "bristle", "orphan": "orphan",
         "chordotonal_organ": "chordotonal_untyped"}
acc = res[res.inferred.isin(EVERY)]
acc = acc[[BANC2.get(k, k) != c for k, c in zip(acc.kind, acc.inferred)]]
relabel = {"basis": "inferred from BANC morphology and wiring (tools/classify_leg_sensory.py); validated rules, precision "
                    + ", ".join(f"{k} {v_:.1%}" for k, v_ in prec.items()) + " on labelled middle/hind neurons",
           "neurons": {str(int(i)): {"class": r.inferred, "banc": r.kind, "leg": r.leg, "side": r.side, "rule": r.rule}
                       for i, r in acc.iterrows()}}
json.dump(relabel, open(ROOT / "data/leg_sensory_relabel.json", "w"), indent=1)
print("validated re-labels written:", acc.groupby(["kind", "inferred"]).size().to_dict())
