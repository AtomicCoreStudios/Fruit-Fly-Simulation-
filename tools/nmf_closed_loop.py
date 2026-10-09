"""Closed loop, offline: nerve-cord rate model -> real leg motor neurons -> joint torques on the physical NeuroMechFly body
(tools/nmf_body.py: MuJoCo/FlyGym 2.1.0, true scale, friction, pad adhesion) -> joint state and cuticular leg load
-> the leg's real proprioceptor neurons -> nerve cord. The live counterpart with the full Godot brain is
tools/body_server.py.

Neural side (tools/vnc_body_model.py): the VNC rate model (Pugliese et al. 2025 equations on our BANC graph,
measured/cable-estimated sizes, 5 ms synapse, 1.8 ms delay; tools/export_vnc_rate_net.py). Descending input: the
tuned DN mix of tools/dn_mix_search.py (DNg100:DNb08:DNa02:DNg97 = 400:100:100:100, basis tuned). Afferents (claw,
hook, club, hair plate, campaniform) are clamped to rates computed from the MuJoCo joint state and the bending moment
carried through each femur base (cuticular strain); hook/club get presynaptic inhibition (Dallmann 2025); hair-plate
direction from tools/hair_plate_tuning.py.

Motor side: per degree of freedom the summed agonist/antagonist motor-unit activations give
    torque = K x lim x tanh(DTHETA_MN x (A_pos - A_neg) / lim)
against the passive joint spring K (springref = NeuroMechFly neutral pose); this reproduces the Godot model's
equilibrium angle when the leg is unloaded.

usage: .venv-flygym/Scripts/python tools/nmf_closed_loop.py [--T 2] [--amps DNg100:400,DNb08:100,DNa02:100,DNg97:100]
       [--proprio 1] [--sensors kinds] [--adhesion pad|auto|on|off] [--pad_fmax 10] [--adh_gain 40] [--cs_muscle 0]
       [--k_joint 10] [--tag x]
-> recordings/nmf_closed_loop<tag>.csv (5 ms samples) and a printed gait summary."""
import sys, time, json, argparse, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from nmf_body import Body, DT_P
from vnc_body_model import Cord, Legs, LEGS, DOFS, KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--amps", default="DNg100:400,DNb08:100,DNa02:100,DNg97:100")
ap.add_argument("--proprio", type=int, default=1)
ap.add_argument("--sensors", default=",".join(KINDS))
ap.add_argument("--adhesion", default="pad", choices=["pad", "auto", "on", "off"])
ap.add_argument("--pad_fmax", type=float, default=10.0)    # uN per leg, assumed (no Drosophila measurement found)
ap.add_argument("--adh_gain", type=float, default=40.0)    # uN, FlyGym actuator modes only (NeuroMechFly v2 default)
ap.add_argument("--cs_muscle", type=float, default=0.0)
ap.add_argument("--k_joint", type=float, default=10.0)     # uN mm / rad, FlyGym default (assumed by FlyGym)
ap.add_argument("--skeleton", default="ALL_BIOLOGICAL")
ap.add_argument("--memory_mb", type=float, default=64.0)   # MuJoCo arena (contacts/constraints)
ap.add_argument("--tag", default="")
args = ap.parse_args()

DT_N = 0.0002                     # neural step (2 MuJoCo steps)
DN_ON = 0.2                       # s of settling before the descending drive starts

cord = Cord(dt=DT_N)
cord.set_drive({kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.amps.split(",") if kv})
legs = Legs(cord, k_joint=args.k_joint, sensors=args.sensors.split(","), cs_muscle=args.cs_muscle)
body = Body(k_joint=args.k_joint, skeleton=args.skeleton, adhesion=args.adhesion, pad_fmax=args.pad_fmax,
            adh_gain=args.adh_gain, memory_mb=args.memory_mb)
data, thor = body.data, body.thor
print(f"skeleton {args.skeleton}: {len(body.jname)} joint DoFs, mass {body.model.body_subtreemass[thor] * 1000:.3f} mg, "
      f"weight {body.W:.2f} uN; trochanter flexion lifts the tarsus in {[l for l, s in zip(LEGS, body.lift_sign) if s > 0]}")

lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"]
swing = [[cord.pos[x] for x in lm[l]["motor"]["ThC_pro"]["pos"] + lm[l]["motor"]["CTr"]["pos"]] for l in LEGS]
stance = [[cord.pos[x] for x in lm[l]["motor"]["ThC_pro"]["neg"] + lm[l]["motor"]["CTr"]["neg"]] for l in LEGS]
rows = []; t0 = time.time(); rates = np.zeros(len(legs.s_idx)); lx = np.zeros(6)
n_sub = int(round(DT_N / DT_P))
for s in range(int(args.T / DT_N)):
    t = s * DT_N
    ang, om, contact, fn = body.state()
    if args.proprio:
        lx = body.leg_load()
        rates = legs.afferents(ang, om, lx)
    cord.step(drive_on=t >= DN_ON, clamp_idx=legs.s_idx if args.proprio else None, clamp_val=rates)
    tq = legs.torques(DT_N)
    lift = (legs.A[:, 0] - legs.A[:, 1]).reshape(6, 6)[:, 2] * body.lift_sign
    body.step(tq, lift, contact, n_sub)
    if s % 25 == 0:      # 5 ms
        th = data.xpos[thor]
        row = {"t": round(t, 4), "x": th[0], "y": th[1], "z": th[2], "up_z": data.xmat[thor][8]}
        for i, leg in enumerate(LEGS):
            row[f"{leg}_contact"] = int(contact[i] or body.adh[i] > 0); row[f"{leg}_fn"] = fn[i]; row[f"{leg}_adh"] = body.adh[i]
            row[f"{leg}_load"] = lx[i]
            row[f"{leg}_swing_mn"] = cord.R[swing[i]].sum(); row[f"{leg}_stance_mn"] = cord.R[stance[i]].sum()
            for di, d in enumerate(DOFS):
                row[f"{leg}_{d}"] = ang[i, di]
        rows.append(row)
    if s % 1000 == 0:
        print(f"t {t:.2f}s  thorax ({data.xpos[thor][0]:+.3f},{data.xpos[thor][1]:+.3f},{data.xpos[thor][2]:.3f}) mm  "
              f"contacts {''.join('1' if c else '0' for c in contact)}  wall {time.time() - t0:.0f}s", flush=True)

# ---- summary ----
out = ROOT / f"recordings/nmf_closed_loop{args.tag}.csv"
with open(out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
tt = np.array([r["t"] for r in rows]); sel = tt >= DN_ON + 0.3
X = np.array([[r["x"], r["y"]] for r in rows]); dur = tt[sel][-1] - tt[sel][0]; disp = X[sel][-1] - X[sel][0]
print(f"\n{args.tag or 'run'}: amps {args.amps or '-'} proprio {args.proprio} sensors {args.sensors} adhesion {args.adhesion}"
      + (f" pad_fmax {args.pad_fmax}" if args.adhesion == "pad" else f" adh_gain {args.adh_gain}"))
if args.proprio:
    print("mean afferent rate by kind at the end (Hz):",
          {k: round(float(rates[legs.s_kind == i].mean()), 1) for i, k in enumerate(KINDS) if (legs.s_kind == i).any()})
upz = np.array([r["up_z"] for r in rows])[sel]
print(f"displacement over the last {dur:.2f} s: forward {disp @ body.fwd[:2]:+.3f} mm, total {np.linalg.norm(disp):.3f} mm "
      f"({np.linalg.norm(disp) / dur:.2f} mm/s); thorax height {np.mean([r['z'] for r, k in zip(rows, sel) if k]):.3f} mm; "
      f"upright (thorax z-axis up) {np.mean(upz > 0.5):.0%} of the time")
for leg in LEGS:
    c = np.array([r[f"{leg}_contact"] for r in rows])[sel]; td = int(np.sum((c[1:] == 1) & (c[:-1] == 0)))
    sw = np.array([r[f"{leg}_swing_mn"] for r in rows])[sel]; st = np.array([r[f"{leg}_stance_mn"] for r in rows])[sel]
    print(f"  {leg}: stance fraction {c.mean():.2f}, touchdowns {td} ({td / dur:.1f}/s), "
          f"ThC_pro range {np.ptp([r[f'{leg}_ThC_pro'] for r, k in zip(rows, sel) if k]):.2f} rad, "
          f"CTr range {np.ptp([r[f'{leg}_CTr'] for r, k in zip(rows, sel) if k]):.2f} rad, swing/stance MN {sw.mean():.0f}/{st.mean():.0f} Hz")
print("saved", out.name, f"| wall {time.time() - t0:.0f} s")
