"""Direction tuning of the leg hair-plate afferents from their network effect.
Hair plates are joint-limit detectors that drive negative feedback (Pratt et al. 2024, Curr Biol, Drosophila coxa and
trochanter hair plates). Joint per neuron: CTr if BANC annotates the hair plate on the trochanter (body_part), else
ThC_pro (coxal hair plates; assumed). Each afferent is driven alone (+80 Hz over the neutral standing state, no
descending drive) through the full VNC rate network (tools/vnc_body_model.py) and the change of its joint's torque
is read; the afferent is then tuned to fire at the limit its output pushes away from:
  dTorque < 0 -> fires at the + limit (protraction / trochanter flexion), dTorque > 0 -> at the - limit.
Effects below 0.02 uN mm are labelled 'none' (fires at either limit, as before). basis: joint from annotation
(measured) or assumed; direction from wiring + limit-detector function (literature-based assumption).
usage: .venv-flygym/Scripts/python tools/hair_plate_tuning.py [legs] -> data/hair_plate_tuning<_legs>.json
       (run without legs to merge the per-leg files into data/hair_plate_tuning.json)"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, Legs, LEGS, DOFS, ROOT

W_BODY, T = 10.05, 0.3
legs = sys.argv[1].split(",") if len(sys.argv) > 1 else None
if legs is None:      # merge
    out = {}
    for f in sorted(ROOT.glob("data/hair_plate_tuning_*.json")):
        out.update(json.load(open(f))); f.unlink()
    json.dump(out, open(ROOT / "data/hair_plate_tuning.json", "w"), indent=0)
    from collections import Counter
    print(len(out), "hair plates:", Counter((v["joint"], v["limit"]) for v in out.values()))
    sys.exit()
part = {}
import csv
with open(ROOT / "data/vnc/vnc_neurons.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["cell_class"] == "hair_plate_neuron":
            part[int(r["model_index"])] = r["body_part"]
lm = json.load(open(ROOT / "data/leg_motor_map.json"))
cord = Cord(); body = Legs(cord)
ang = np.zeros((6, 6)); om = np.zeros((6, 6)); fn = np.full(6, W_BODY / 6)


def settle(extra_idx=None):
    cord.reset(); body.act[:] = 0; tq = []
    for s in range(int(T / cord.dt)):
        r = body.afferents(ang, om, 3.0 * fn / W_BODY)
        if extra_idx is not None:
            r = r.copy(); r[extra_idx] += 80.0
        cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=r)
        t = body.torques(cord.dt)
        if s * cord.dt > T - 0.1:
            tq.append(t)
    return np.mean(tq, 0).reshape(6, 6)


base = settle(); res = {}
for leg in legs:
    li = LEGS.index(leg)
    for x in lm["legs"][leg]["sensory"]["hair_plate"]:
        x = int(x); k = int(np.where(body.s_idx == cord.pos[x])[0][0])
        joint = "CTr" if "trochanter" in part.get(x, "") else "ThC_pro"
        d = settle(k)[li] - base[li]; dj = float(d[DOFS.index(joint)])
        lim = "pos" if dj < -0.02 else ("neg" if dj > 0.02 else "none")
        res[str(x)] = {"leg": leg, "joint": joint, "limit": lim, "dtorque": round(dj, 3),
                       "joint_basis": "annotated" if joint == "CTr" else "assumed"}
        print(leg, x, joint, lim, round(dj, 3), flush=True)
json.dump(res, open(ROOT / f"data/hair_plate_tuning_{'_'.join(legs)}.json", "w"), indent=0)
