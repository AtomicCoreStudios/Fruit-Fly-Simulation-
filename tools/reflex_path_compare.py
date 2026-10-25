"""Offline counterpart of the live 'reflex capacity' test: the left-front flexion-tuned FeCO claw/hook afferents are
clamped at 100 Hz (all other afferents 0, no descending drive) in the offline Pugliese-equation nerve cord, and the
steady rates of the path interneurons and tibia motor pools are printed, for a stage-by-stage comparison with the live
model (recordings/reflex_capacity_flex.csv).
usage: .venv-flygym/Scripts/python tools/reflex_path_compare.py"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, ROOT

lm = json.load(open(ROOT / "data/leg_motor_map.json"))
flex = [int(x) for x in open(ROOT / "recordings/lf_feco_flex_ids.txt").read().split(",")]
cord = Cord()
watch = {"INXXX464": [154028], "IN20A.22A004": [151368], "IN20A.22A005": [144183], "IN19A005": [150800],
         "lf_ext": lm["legs"]["lf"]["motor"]["FTi"]["neg"], "lf_flex": lm["legs"]["lf"]["motor"]["FTi"]["pos"]}
sens_all = np.array(sorted({cord.pos[int(x)] for L in lm["legs"].values() for v in L["sensory"].values() for x in v}))
for label, drive in (("none", []), ("flex 100 Hz", flex)):
    cord.reset()
    val = np.zeros(len(sens_all)); idx = {x: i for i, x in enumerate(sens_all)}
    for x in drive:
        val[idx[cord.pos[x]]] = 100.0
    acc = {k: 0.0 for k in watch}; n = 0
    for s in range(int(0.6 / cord.dt)):
        cord.step(drive_on=False, clamp_idx=sens_all, clamp_val=val)
        if s * cord.dt > 0.3:
            for k, ids in watch.items():
                acc[k] += cord.R[[cord.pos[i] for i in ids]].mean()
            n += 1
    print(label.ljust(12), " ".join(f"{k} {acc[k] / n:.1f}" for k in watch))
