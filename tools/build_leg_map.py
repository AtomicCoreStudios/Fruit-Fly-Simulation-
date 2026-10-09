"""Leg neuromuscular map -> data/leg_motor_map.json for scripts/fly_legs.gd.
Motor side: every BANC leg motor neuron (data/vnc/vnc_neurons.csv) is assigned to a joint degree of freedom and
direction from its BANC 'function' annotation (muscle-target based; cf. Azevedo et al. 2024 Nature for the
motor-neuron-to-muscle map of the fly leg). Sensory side: the leg's proprioceptors by class (femoral chordotonal
organ claw = position, hook = movement direction, club = vibration/bidirectional movement; hair plates = joint
angle; campaniform sensilla = cuticular strain/load; Tuthill & Wilson 2016 Curr Biol review; Mamiya et al. 2018).
DoF sign convention (+ / -):
  ThC_pro  coxa protraction (+, move_coxa_anterior) / retraction (-, move_coxa_posterior[_lateral])
  ThC_add  coxa adduction (+, move_coxa_medial) / abduction (-, move_coxa_posterior_lateral, half weight)
  CTr      trochanter flexion (+, flex_coxa_trochanter_joint) / extension (-, extend_coxa_trochanter_joint)
  FeRot    femur reduction/rotation (+, femur_reductor 'unknown_leg_movement'; passive return)
  FTi      tibia flexion (+, flex_femur_tibia_joint) / extension (-, extend_femur_tibia_joint)
  TiTa     tarsus depression (+, extend_tibia_tarsus_joint, pull_long_tendon) / levation (-, flex_tibia_tarsus_joint)"""
import json, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
LEG = {"front": "f", "middle": "m", "hind": "h"}
ACT = {"move_coxa_anterior": [("ThC_pro", +1, 1.0)], "move_coxa_posterior": [("ThC_pro", -1, 1.0)],
       "move_coxa_posterior_lateral": [("ThC_pro", -1, 0.5), ("ThC_add", -1, 0.5)],
       "move_coxa_medial": [("ThC_add", +1, 1.0)],
       "flex_coxa_trochanter_joint": [("CTr", +1, 1.0)], "extend_coxa_trochanter_joint": [("CTr", -1, 1.0)],
       "unknown_leg_movement": [("FeRot", +1, 1.0)],
       "flex_femur_tibia_joint": [("FTi", +1, 1.0)], "extend_femur_tibia_joint": [("FTi", -1, 1.0)],
       "extend_tibia_tarsus_joint": [("TiTa", +1, 1.0)], "pull_long_tendon": [("TiTa", +1, 1.0)],
       "flex_tibia_tarsus_joint": [("TiTa", -1, 1.0)]}
out = {"basis": __doc__.split("\n")[1], "legs": {}}
lm = v[v.cell_class.astype(str).str.contains("leg_motor") & v.sub_class.astype(str).str.contains("_leg_motor")]
unmapped = 0
for r in lm.itertuples():
    leg = ("l" if r.side == "left" else "r") + LEG[r.sub_class.split("_")[0]]
    acts = [a for a in str(r.function).split(",") if a not in ("leg_motor", "jump_escape")]
    m = [x for a in acts for x in ACT.get(a, [])]
    if not m:
        unmapped += 1; continue
    L = out["legs"].setdefault(leg, {"motor": {}, "sensory": {}})
    for dof, sgn, w in m:
        L["motor"].setdefault(dof, {"pos": [], "neg": [], "w_pos": [], "w_neg": []})
        key = "pos" if sgn > 0 else "neg"
        L["motor"][dof][key].append(int(r.model_index)); L["motor"][dof]["w_" + key].append(w)
# motor-unit force ~ motor-neuron size (Henneman size principle; shown for Drosophila leg motor neurons by Azevedo et al.
# 2020 eLife: small, early-recruited neurons give small forces, large ones large forces). Each neuron's weight in its
# pool is scaled by its surface area (predicted from BANC skeleton cable length, r = 0.947; pool median where missing)
# divided by the pool mean, so the pool's total force is unchanged. basis: principle literature, sizes measured,
# proportionality assumed. FLY_MN_FORCE_UNIFORM=1: equal force per motor neuron (before 2026-10-09).
import os as _os, numpy as _np
if _os.environ.get("FLY_MN_FORCE_UNIFORM") != "1":
    _cab = pd.read_csv(ROOT / "data/vnc/banc_cable_length.csv").set_index("root_id").cable_um
    _bid = v.set_index("model_index").bid
    for _leg, _L in out["legs"].items():
        for _dof, _m in _L["motor"].items():
            for _k in ("pos", "neg"):
                _a = _np.array([_np.exp(1.179 + 1.037 * _np.log(_cab.get(_bid.get(i), _np.nan))) for i in _m[_k]], float)
                if len(_a) == 0:
                    continue
                _a = _np.where(_np.isfinite(_a), _a, _np.nanmedian(_a) if _np.isfinite(_a).any() else 1.0)
                _m["w_" + _k] = [round(float(w * x / _a.mean()), 4) for w, x in zip(_m["w_" + _k], _a)]
    out["motor_force_basis"] = "size principle: weight x area / pool mean area (Azevedo et al. 2020); FLY_MN_FORCE_UNIFORM=1 for equal force"
SENS = {"claw_chordotonal": "FeCO_claw", "hook_chordotonal": "FeCO_hook", "club_chordotonal": "FeCO_club",
        "hair_plate": "hair_plate", "campaniform": "campaniform"}
# validated re-labels of leg sensory neurons from morphology + wiring (tools/classify_leg_sensory.py, ~95% precision):
# unlabelled / untyped / mislabelled neurons identified as FeCO claw/hook/club join those lists; neurons identified as
# bristles, taste or multidendritic leave the proprioceptor lists (kept in data/leg_sensory_relabel.json for later)
import json as _json
_rl = (ROOT / "data/leg_sensory_relabel.json")
RELABEL = {int(k): x["class"] for k, x in _json.load(open(_rl))["neurons"].items()} if _rl.exists() else {}
sn = v[v.super_class.isin(["sensory", "sensory_ascending"]) & v.body_part.astype(str).str.contains("leg")]
for r in sn.itertuples():
    k = next((lab for key, lab in SENS.items() if key in str(r.sub_class)), None)
    if int(r.model_index) in RELABEL:
        k = RELABEL[int(r.model_index)] if RELABEL[int(r.model_index)] in ("FeCO_claw", "FeCO_hook", "FeCO_club") else None
    part = str(r.body_part).split("_")[0]
    if k is None or part not in LEG or r.side not in ("left", "right"):
        continue
    leg = ("l" if r.side == "left" else "r") + LEG[part]
    out["legs"].setdefault(leg, {"motor": {}, "sensory": {}})["sensory"].setdefault(k, []).append(int(r.model_index))
# FeCO claw/hook tuning from wiring (Lee et al. 2025 Nat Commun): flexion-encoding axons excite tibia EXTENSOR and
# inhibit tibia flexor motor neurons; extension-encoding axons the reverse. Score = signed 1-hop + 2-hop drive onto
# the leg's FTi extensor pool minus its FTi flexor pool; > 0 -> "flex", < 0 -> "ext". basis: measured wiring +
# published criterion
import numpy as np, scipy.sparse as sp
e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet")
ids = np.unique(np.concatenate([e.Presynaptic_Index, e.Postsynaptic_Index])); pos = {x: i for i, x in enumerate(ids)}
W = sp.csr_matrix((e["Excitatory x Connectivity"].astype(float).values,
                   ([pos[x] for x in e.Postsynaptic_Index], [pos[x] for x in e.Presynaptic_Index])), shape=(len(ids), len(ids)))
Wn = W.multiply(1.0 / np.maximum(np.abs(W).sum(axis=1), 1.0)).tocsr()      # row-normalised (fraction of input)
tuning = {}
for leg, L in out["legs"].items():
    m = L["motor"].get("FTi", {}); ext = [pos[x] for x in m.get("neg", []) if x in pos]; flx = [pos[x] for x in m.get("pos", []) if x in pos]
    for kind in ("FeCO_claw", "FeCO_hook"):
        for s_ in L["sensory"].get(kind, []):
            if s_ not in pos:
                continue
            v0 = np.zeros(len(ids)); v0[pos[s_]] = 1.0
            h1 = Wn @ v0; h2 = Wn @ np.maximum(h1, 0.0)            # via excited interneurons (sign kept)
            sc = (h1[ext].sum() + h2[ext].sum()) - (h1[flx].sum() + h2[flx].sum())
            tuning[str(s_)] = "flex" if sc > 0 else ("ext" if sc < 0 else "none")
out["feco_tuning"] = tuning
# presynaptic inhibition of movement-encoding afferents (hook, club; Dallmann et al. 2025 Nature): the inhibitory
# (GABA/Glu) VNC neurons that synapse onto each axon terminal, with synapse counts (BANC). These synapses are not
# in the spiking graph (afferent inputs are removed in build_connectome); fly_legs.gd uses them to scale release.
pre_inh = {}
neg = e[(e["Excitatory"] < 0)]
for leg, L in out["legs"].items():
    for kind in ("FeCO_hook", "FeCO_club"):
        for s_ in L["sensory"].get(kind, []):
            q = neg[neg.Postsynaptic_Index == s_]
            if len(q):
                pre_inh[str(s_)] = [[int(a), int(b)] for a, b in zip(q.Presynaptic_Index, q.Connectivity)]
out["presyn_inhibition"] = pre_inh
(ROOT / "data/leg_motor_map.json").write_text(json.dumps(out, indent=0))
from collections import Counter
print("FeCO tuning from wiring:", Counter(tuning.values()))
print("movement afferents with presynaptic inhibitory input:", len(pre_inh), "| median synapses",
      int(np.median([sum(b for _, b in v) for v in pre_inh.values()])) if pre_inh else 0)
for leg, L in sorted(out["legs"].items()):
    print(leg, " ".join(f"{d}:{len(x['pos'])}/{len(x['neg'])}" for d, x in sorted(L["motor"].items())),
          "| " + " ".join(f"{k}:{len(x)}" for k, x in sorted(L["sensory"].items())))
print("unmapped leg MNs:", unmapped)
