"""Where does the extensor drive come from in the OFFLINE reflex protocol (tools/reflex_offline.py, k = 1)?
Runs the protocol once and reports, during the imposed movement: mean rate per neuron class, and the excitatory /
inhibitory synaptic drive (synapses x Hz, via the cord's weight matrix) onto the left-front tibia extensor pool by
presynaptic class. Used to compare the offline (all rate units) with the live model (ANs and other non-leg nerve-cord
neurons spiking).
usage: .venv-flygym/Scripts/python tools/reflex_offline_sources.py"""
import sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vnc_body_model import Cord, Legs, LEGS, ROOT

li = LEGS.index("lf")
lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"]["lf"]["motor"]["FTi"]
cord = Cord(); body = Legs(cord)
ext = np.array([cord.pos[x] for x in lm["neg"]])
cls = {}
with open(ROOT / "data/vnc/vnc_neurons.csv", newline="") as f:
    for r in csv.DictReader(f):
        cls[int(r["model_index"])] = r["super_class"]
klass = np.array([cls.get(int(x), "brain_bridged") for x in cord.ids])
AMP, F, T_SETTLE, N_CYC = 0.1745, 2.0, 0.5, 2
ang = np.zeros((6, 6)); om = np.zeros((6, 6)); lx = np.full(6, 0.5)
acc = np.zeros(cord.n); nacc = 0
for s in range(int((T_SETTLE + N_CYC / F) / cord.dt)):
    t = s * cord.dt
    if t >= T_SETTLE:
        ph = 2 * np.pi * F * (t - T_SETTLE)
        ang[li, 4] = AMP * np.sin(ph); om[li, 4] = AMP * 2 * np.pi * F * np.cos(ph)
    cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=body.afferents(ang, om, lx))
    if t >= T_SETTLE:
        acc += cord.R; nacc += 1
R = acc / nacc
print("extensor pool mean rate during movement:", round(float(R[ext].mean()), 1), "Hz")
for k in sorted(set(klass)):
    m = klass == k
    print(f"  class {k:32s} n {m.sum():6d}  mean rate {R[m].mean():7.2f} Hz  active(>5 Hz) {int((R[m] > 5).sum())}")
Wd = cord.W[ext].toarray()            # rows: extensor MNs, cols: presynaptic (weights 0.03 x signed synapses)
contrib = Wd * R[None, :]
print("drive onto the extensor pool by presynaptic class (Pugliese input units):")
for k in sorted(set(klass)):
    m = klass == k
    c = contrib[:, m]
    print(f"  {k:32s} excit {c.clip(min=0).sum():9.1f}  inhib {c.clip(max=0).sum():9.1f}")

# top individual contributors (for live comparison) -> recordings/reflex_offline_top_contributors.csv
tot = contrib.sum(0); order = np.argsort(tot)
rows = [(int(cord.ids[j]), float(tot[j]), float(R[j]), klass[j]) for j in list(order[-15:][::-1]) + list(order[:8])]
with open(ROOT / "recordings/reflex_offline_top_contributors.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model_index", "drive_units", "offline_rate_hz", "class"]); w.writerows(rows)
for r in rows: print("  ", r)
