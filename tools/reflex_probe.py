"""Reflex sign probe of the nerve cord (no body, no descending drive): impose a joint state on one leg, let the cord
settle with the afferents clamped to the rates of tools/vnc_body_model.py, and read the joint torque that the leg's
motor neurons produce. A stabilising (resistance) reflex opposes the imposed change: torque change < 0 for an imposed
flexion/protraction (+), as measured for the femur-tibia joint in flies (FeCO-driven resistance reflex; Bassler 1993
in stick insects, Agrawal et al. 2020 / Azevedo et al. 2020 in Drosophila). Load: campaniform input from an extra
half body weight on the leg (stance reinforcement expected: Zill et al. 2004).
usage: .venv-flygym/Scripts/python tools/reflex_probe.py [legs=lf,lm,lh,rf,rm,rh] [settle_s=0.3]
-> recordings/reflex_probe.csv"""
import sys, pathlib, csv
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, Legs, LEGS, DOFS, ROOT

legs = sys.argv[1].split(",") if len(sys.argv) > 1 else LEGS
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
W_BODY = 10.05          # uN, NeuroMechFly mass 1.024 mg x g
cord = Cord(); body = Legs(cord)
CONDS = [("neutral", None, 0, 0), ("FTi +0.3 rad", "FTi", 0.3, 0), ("FTi -0.3 rad", "FTi", -0.3, 0),
         ("FTi +3 rad/s", "FTi", 0, 3.0), ("FTi -3 rad/s", "FTi", 0, -3.0),
         ("ThC_pro +0.4 rad", "ThC_pro", 0.4, 0), ("CTr +0.5 rad", "CTr", 0.5, 0), ("load +W/2", "load", 0, 0)]


def settle(li, dof, x, v):
    cord.reset(); body.act[:] = 0
    ang = np.zeros((6, 6)); om = np.zeros((6, 6)); fn = np.full(6, W_BODY / 6)
    if dof in DOFS:
        ang[li, DOFS.index(dof)] = x; om[li, DOFS.index(dof)] = v
    if dof == "load":
        fn[li] += W_BODY / 2
    tq = []
    for s in range(int(T / cord.dt)):
        cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=body.afferents(ang, om, fn, W_BODY))
        t = body.torques(cord.dt)
        if s * cord.dt > T - 0.1:
            tq.append(t)
    return np.mean(tq, 0).reshape(6, 6)


rows = []
for leg in legs:
    li = LEGS.index(leg); base = settle(li, None, 0, 0)
    for name, dof, x, v in CONDS[1:]:
        tq = settle(li, dof, x, v); d = tq[li] - base[li]
        row = {"leg": leg, "condition": name, **{f"d_{k}": round(float(d[i]), 3) for i, k in enumerate(DOFS)},
               **{f"base_{k}": round(float(base[li][i]), 3) for i, k in enumerate(DOFS)}}
        if dof in DOFS:
            sgn = np.sign(x if x else v); row["verdict"] = "resist" if d[DOFS.index(dof)] * sgn < -0.05 else (
                "assist" if d[DOFS.index(dof)] * sgn > 0.05 else "none")
        else:
            row["verdict"] = f"FTi {d[4]:+.2f} CTr {d[2]:+.2f} ThC_pro {d[0]:+.2f}"
        rows.append(row)
        print(f"{leg} {name:16s} dTorque (uN mm) " + " ".join(f"{k} {d[i]:+.2f}" for i, k in enumerate(DOFS)) + f"  -> {row['verdict']}", flush=True)
    print(f"{leg} baseline torque at neutral, no drive: " + " ".join(f"{k} {base[li][i]:+.2f}" for i, k in enumerate(DOFS)), flush=True)
with open(ROOT / "recordings/reflex_probe.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
