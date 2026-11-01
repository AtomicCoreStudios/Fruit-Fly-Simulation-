"""How many leg motor pools does the DNg100-driven rhythm reach in the OFFLINE nerve cord (Pugliese equations)?
Conditions (any combination in one label): none, rest (live size-dependent MN resting offsets from data/physiology.bin),
homeo (live homeostatic offsets, capped), adapt (live rate-unit adaptation). Reports pools with s.d. > 2 Hz.
usage: .venv-flygym/Scripts/python tools/cpg_reach_offline.py <DNg100 input> none rest rest+homeo+adapt ..."""
import sys, json, numpy as np
sys.path.insert(0, 'tools')
from vnc_body_model import Cord, ROOT
ph = np.fromfile(ROOT/'data/physiology.bin', np.float32).reshape(-1,4)
lm = json.load(open(ROOT/'data/leg_motor_map.json'))['legs']
pools = {f'{l}_{d}_{k}': [x for x in L['motor'][d][k]] for l,L in lm.items() for d in ['CTr','ThC_pro','FTi'] for k in ['pos','neg']}
drive = float(sys.argv[1])
for label in sys.argv[2:]:
    c = Cord()
    c.use_live_physiology(adapt='adapt' in label, offsets=(ROOT/'data/homeostasis_offsets.bin') if 'homeo' in label else None)
    if 'rest' in label:
        for P in pools.values():
            for x in P:
                if x in c.pos: c.offset[c.pos[x]] += 21.82 * ph[x,0]
    c.set_drive({'DNg100': drive})
    rec = []
    for s in range(int(2.0/c.dt)):
        c.step(drive_on=True)
        if s*c.dt > 0.5 and s % 5 == 0:
            rec.append([c.R[[c.pos[x] for x in P if x in c.pos]].sum() for P in pools.values()])
    rec = np.array(rec)
    sd = rec.std(0); m = rec.mean(0)
    print(f'{label:20s} rhythmic pools (sd>2 Hz) {int((sd>2).sum()):2d}/36, mean pool rate {m.mean():.1f} Hz, total sd {sd.sum():.0f}', flush=True)
