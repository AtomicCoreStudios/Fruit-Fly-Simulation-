"""Which afferent kinds set each leg's resting motor tone? Neutral pose, equal load share (W/6 per leg), no
descending drive; settle the nerve cord with only one afferent kind active (or all) and print the resting joint
torques per leg (uN mm, biological sign: + protract / adduct / flex / lower). Used to trace left/right asymmetries.
usage: .venv-flygym/Scripts/python tools/tone_asymmetry.py [kinds,...]"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, Legs, LEGS, DOFS, KINDS

T = 0.3
sets = [k.split("+") for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else [[k] for k in KINDS] + [KINDS]
cord = Cord()
for ks in sets:
    body = Legs(cord, sensors=ks); cord.reset()
    ang = np.zeros((6, 6)); om = np.zeros((6, 6)); lx = np.full(6, 0.5)
    tq = []
    for s in range(int(T / cord.dt)):
        cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=body.afferents(ang, om, lx))
        t = body.torques(cord.dt)
        if s * cord.dt > T - 0.1:
            tq.append(t)
    tq = np.mean(tq, 0).reshape(6, 6)
    print("+".join(ks) if len(ks) < 5 else "ALL")
    for li, leg in enumerate(LEGS):
        print(f"   {leg} " + " ".join(f"{d} {tq[li, di]:+6.2f}" for di, d in enumerate(DOFS)))
