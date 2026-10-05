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
SENS = {"claw_chordotonal": "FeCO_claw", "hook_chordotonal": "FeCO_hook", "club_chordotonal": "FeCO_club",
        "hair_plate": "hair_plate", "campaniform": "campaniform"}
sn = v[v.super_class.isin(["sensory", "sensory_ascending"]) & v.body_part.astype(str).str.contains("leg")]
for r in sn.itertuples():
    k = next((lab for key, lab in SENS.items() if key in str(r.sub_class)), None)
    part = str(r.body_part).split("_")[0]
    if k is None or part not in LEG or r.side not in ("left", "right"):
        continue
    leg = ("l" if r.side == "left" else "r") + LEG[part]
    out["legs"].setdefault(leg, {"motor": {}, "sensory": {}})["sensory"].setdefault(k, []).append(int(r.model_index))
(ROOT / "data/leg_motor_map.json").write_text(json.dumps(out, indent=0))
for leg, L in sorted(out["legs"].items()):
    print(leg, " ".join(f"{d}:{len(x['pos'])}/{len(x['neg'])}" for d, x in sorted(L["motor"].items())),
          "| " + " ".join(f"{k}:{len(x)}" for k, x in sorted(L["sensory"].items())))
print("unmapped leg MNs:", unmapped)
