"""Resistance-reflex protocol (Akitake et al. 2015: tibia moved ~20 deg peak-to-peak at 2 Hz, decapitated, femur fixed;
tibial extensor EMG 107 +- 12 spikes per cycle) in the OFFLINE nerve-cord model: Pugliese et al. 2025 rate equations
on our BANC graph (tools/vnc_body_model.py Cord; no homeostasis, no adaptation, no brain input), with the leg
proprioceptors computed from the imposed joint state (other legs neutral, equal load share).
Reports tibia extensor and flexor pool spikes per cycle for a range of afferent synaptic strengths (release gain k,
applied to the afferent rates), to test whether the published model on this wiring can reach the measured reflex.
usage: .venv-flygym/Scripts/python tools/reflex_offline.py [leg=lf] -> recordings/reflex_offline.csv"""
import sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, Legs, LEGS, DOFS, ROOT

leg = sys.argv[1] if len(sys.argv) > 1 else "lf"; li = LEGS.index(leg)
lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"][leg]["motor"]["FTi"]
cord = Cord(); body = Legs(cord)
ext = [cord.pos[x] for x in lm["neg"]]; flx = [cord.pos[x] for x in lm["pos"]]
AMP, F, T_SETTLE, N_CYC = 0.1745, 2.0, 0.5, 3
rows = []
for k in [1, 2, 4, 8, 16, 32]:
    cord.reset(); body.act[:] = 0
    ang = np.zeros((6, 6)); om = np.zeros((6, 6)); lx = np.full(6, 0.5)
    se, sf, n = 0.0, 0.0, 0; rest, efl, eex, nfl, nex = 0.0, 0.0, 0.0, 0, 0
    for s in range(int((T_SETTLE + N_CYC / F) / cord.dt)):
        t = s * cord.dt
        if t >= T_SETTLE:
            ph = 2 * np.pi * F * (t - T_SETTLE)
            ang[li, 4] = AMP * np.sin(ph); om[li, 4] = AMP * 2 * np.pi * F * np.cos(ph)
        cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=k * body.afferents(ang, om, lx))
        if t >= T_SETTLE:
            se += cord.R[ext].sum() * cord.dt; sf += cord.R[flx].sum() * cord.dt
            if ang[li, 4] > 0: efl += cord.R[ext].sum(); nfl += 1
            else: eex += cord.R[ext].sum(); nex += 1
        elif t >= T_SETTLE - 0.2:
            rest += cord.R[ext].sum() * cord.dt / 0.2 * 0.5      # spikes per 0.5 s (one cycle) at rest
    rows.append({"k": k, "extensor_spikes_per_cycle": round(se / N_CYC, 1), "extensor_rest_per_cycle": round(rest, 1),
                 "extensor_Hz_flexion_half": round(efl / max(nfl, 1), 1), "extensor_Hz_extension_half": round(eex / max(nex, 1), 1),
                 "flexor_spikes_per_cycle": round(sf / N_CYC, 1)})
    print(rows[-1], flush=True)
with open(ROOT / "recordings/reflex_offline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
