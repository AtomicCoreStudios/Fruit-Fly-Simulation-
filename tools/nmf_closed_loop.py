"""Closed loop: nerve-cord rate model -> real leg motor neurons -> joint torques on the NeuroMechFly body in MuJoCo
(FlyGym 2.1.0, true scale: mm, g, s; 1.02 mg fly; contact physics with friction and tarsal adhesion) -> joint state
and ground reaction forces -> the leg's real proprioceptor neurons -> nerve cord.

Neural side (tools/vnc_body_model.py): the VNC rate model (Pugliese et al. 2025 equations on our BANC graph,
measured/cable-estimated sizes, 5 ms synapse, 1.8 ms delay; tools/export_vnc_rate_net.py). Descending input: the
tuned DN mix of tools/dn_mix_search.py (DNg100:DNb08:DNa02:DNg97 = 400:100:100:100, basis tuned). Afferents (claw,
hook, club, hair plate, campaniform) are clamped to rates computed from the MuJoCo joint state and the bending
moment carried through each femur base (cuticular strain); hook/club get presynaptic inhibition (Dallmann 2025); hair-plate direction from
tools/hair_plate_tuning.py.

Body side: per degree of freedom the summed agonist/antagonist motor-unit activations give
    torque = K_JOINT x lim x tanh(DTHETA_MN x (A_pos - A_neg) / lim)
which, against the passive joint spring K_JOINT (springref = NeuroMechFly neutral pose), reproduces the Godot model's
equilibrium angle when the leg is unloaded; with the foot on the ground it works against body weight, friction and
adhesion. Coxa yaw, tarsal, head, abdomen joints are passive springs. Adhesion is on while a tarsus touches the
ground and the leg's trochanter drive is not lifting it (assumed: pulvilli peel off when the leg is actively levated).
The full biological skeleton is used so the head (eyes) is a physical body moved by the physics.

usage: .venv-flygym/Scripts/python tools/nmf_closed_loop.py [--T 2] [--amps DNg100:400,DNb08:100,DNa02:100,DNg97:100]
       [--proprio 1] [--sensors kinds] [--adhesion auto|on|off] [--cs_muscle 0] [--k_joint 10] [--tag x]
-> recordings/nmf_closed_loop<tag>.csv (5 ms samples) and a printed gait summary."""
import os, sys, time, json, argparse, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
# keep FlyGym's lazily downloaded assets inside the project (machine rule: no writes outside the project folder)
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(ROOT / "assets/flygym_cache"))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import mujoco as mj
import flygym
import flygym.anatomy as A
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D
from vnc_body_model import Cord, Legs, LEGS, DOFS, KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--amps", default="DNg100:400,DNb08:100,DNa02:100,DNg97:100")
ap.add_argument("--proprio", type=int, default=1)
ap.add_argument("--sensors", default=",".join(KINDS))
ap.add_argument("--adhesion", default="auto", choices=["auto", "on", "off"])
ap.add_argument("--cs_muscle", type=float, default=0.0)
ap.add_argument("--adh_gain", type=float, default=40.0)
ap.add_argument("--k_joint", type=float, default=10.0)
ap.add_argument("--skeleton", default="ALL_BIOLOGICAL")
ap.add_argument("--tag", default="")
args = ap.parse_args()

MJ_DOF = {"ThC_pro": "c_thorax-{l}_coxa-pitch", "ThC_add": "c_thorax-{l}_coxa-roll",
          "CTr": "{l}_coxa-{l}_trochanterfemur-pitch", "FeRot": "{l}_coxa-{l}_trochanterfemur-roll",
          "FTi": "{l}_trochanterfemur-{l}_tibia-pitch", "TiTa": "{l}_tibia-{l}_tarsus1-pitch"}
K_JOINT = args.k_joint            # uN mm / rad, FlyGym default joint stiffness (assumed by FlyGym)
D_JOINT = K_JOINT * 0.030         # damping: unloaded joint relaxes with TAU_JOINT = 30 ms (approximate)
ADH_GAIN = args.adh_gain          # uN per leg; 40 = NeuroMechFly v2 default (Wang-Chen et al. 2024). MuJoCo adhesion is a
                                  # constant pull while in contact (real pads resist detachment); no Drosophila value found
                                  # (stick-insect arolia: up to 0.8 x body weight, Labonte/Federle)
DT_N, DT_P = 0.0002, 0.0001       # neural step; MuJoCo step (FlyGym default)
DN_ON = 0.2                       # s of settling before the descending drive starts

cord = Cord(dt=DT_N)
cord.set_drive({kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.amps.split(",") if kv})
legs = Legs(cord, k_joint=K_JOINT, sensors=args.sensors.split(","), cs_muscle=args.cs_muscle)

# ---- body ----
fly = NeuroMechFly()
skel = A.Skeleton(axis_order=A.AxisOrder.PITCH_ROLL_YAW, joint_preset=A.JointPreset[args.skeleton])
fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL, stiffness=K_JOINT, damping=D_JOINT)
jorder = fly.get_jointdofs_order(); jname = [j.name for j in jorder]
act_dofs = [jorder[jname.index(MJ_DOF[d].format(l=leg))] for leg in LEGS for d in DOFS]
fly.add_actuators(act_dofs, ActuatorType.MOTOR, forcerange=(-30.0, 30.0))
fly.add_leg_adhesion(gain=ADH_GAIN)
world = FlatGroundWorld()
world.add_fly(fly, np.array([0.0, 0.0, 0.7]), Rotation3D("quat", [1, 0, 0, 0]), add_ground_contact_sensors=True)
sim = flygym.Simulation(world, timestep=DT_P)
sim.reset()
model, data = sim.mj_model, sim.mj_data
legs_order = list(fly.get_legs_order()); leg_perm = [legs_order.index(l) for l in LEGS]
q_slot = np.array([jname.index(MJ_DOF[d].format(l=leg)) for leg in LEGS for d in DOFS])
q0 = np.array(sim.get_joint_angles(fly.name)).copy()      # neutral pose after reset


def body_id(part):
    for i in range(model.nbody):
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "").endswith(part):
            return i
    raise KeyError(part)


def joint_qadr(name):
    for j in range(model.njnt):
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith(name):
            return model.jnt_qposadr[j]
    raise KeyError(name)


# signs: + must protract, adduct, flex (CTr, FTi), lower the tarsus (TiTa), as in fly_legs.gd _fix_sign
mj.mj_kinematics(model, data)
thor = body_id("c_thorax")
fwd = data.xmat[thor].reshape(3, 3)[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd); up = np.array([0, 0, 1.0])
sign = np.ones(36); lift_sign = np.ones(6); qpos_save = data.qpos.copy()
for li, leg in enumerate(LEGS):
    tip, cox, fem = body_id(f"{leg}_tarsus5"), body_id(f"{leg}_coxa"), body_id(f"{leg}_trochanterfemur")
    mj.mj_kinematics(model, data); p0 = data.xpos[tip].copy(); c0 = data.xpos[cox].copy(); f0 = data.xpos[fem].copy()
    lat = c0 - data.xpos[thor]; lat[2] = 0; lat /= np.linalg.norm(lat)
    for di, d in enumerate(DOFS):
        data.qpos[:] = qpos_save; data.qpos[joint_qadr(MJ_DOF[d].format(l=leg))] += 0.05; mj.mj_kinematics(model, data)
        dp = data.xpos[tip] - p0
        metric = {"ThC_pro": dp @ fwd, "ThC_add": -(dp @ lat) - (dp @ up),
                  "CTr": np.linalg.norm(p0 - c0) - np.linalg.norm(data.xpos[tip] - c0),
                  "FTi": np.linalg.norm(p0 - f0) - np.linalg.norm(data.xpos[tip] - f0),
                  "TiTa": -(dp @ up), "FeRot": 1.0}[d]
        sign[li * 6 + di] = 1.0 if metric >= 0 else -1.0
        if d == "CTr":
            lift_sign[li] = 1.0 if (dp @ up) * sign[li * 6 + di] > 0 else -1.0   # + CTr drive raises the tip?
data.qpos[:] = qpos_save; mj.mj_forward(model, data)
W_BODY = float(model.body_subtreemass[thor]) * 9810.0          # uN
# cuticular load: bending moment carried through each femur base (MuJoCo cfrc_int = force/torque the coxa exerts on the
# femur subtree: weight, contact, adhesion, inertia), perpendicular to the femur axis, in units of
# M_REF = (W/3) x leg length (femur base -> tarsus tip at the neutral pose)
fem_b = [body_id(f"{l}_trochanterfemur") for l in LEGS]; tib_b = [body_id(f"{l}_tibia") for l in LEGS]
tip_b = [body_id(f"{l}_tarsus5") for l in LEGS]
M_REF = np.array([W_BODY / 3 * np.linalg.norm(data.xpos[t] - data.xpos[f]) for f, t in zip(fem_b, tip_b)])


def leg_load():
    mj.mj_rnePostConstraint(model, data)
    out = np.zeros(6)
    for i, (b, tb) in enumerate(zip(fem_b, tib_b)):
        c = data.subtree_com[model.body_rootid[b]]; tq, f = data.cfrc_int[b, :3], data.cfrc_int[b, 3:]
        tp = tq + np.cross(c - data.xpos[b], f)                 # moment about the femur base
        ax = data.xpos[tb] - data.xpos[b]; ax /= np.linalg.norm(ax)
        out[i] = np.linalg.norm(tp - (tp @ ax) * ax) / M_REF[i]
    return out

print(f"skeleton {args.skeleton}: {len(jname)} joint DoFs, mass {model.body_subtreemass[thor] * 1000:.3f} mg, "
      f"weight {W_BODY:.2f} uN; trochanter flexion lifts the tarsus in {[l for l, s in zip(LEGS, lift_sign) if s > 0]}")

# ---- loop ----
lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"]
swing = [[cord.pos[x] for x in lm[l]["motor"]["ThC_pro"]["pos"] + lm[l]["motor"]["CTr"]["pos"]] for l in LEGS]
stance = [[cord.pos[x] for x in lm[l]["motor"]["ThC_pro"]["neg"] + lm[l]["motor"]["CTr"]["neg"]] for l in LEGS]
rows = []; t0 = time.time(); adh = np.ones(6); rates = np.zeros(len(legs.s_idx))
for s in range(int(args.T / DT_N)):
    t = s * DT_N
    qa = np.array(sim.get_joint_angles(fly.name)); qv = np.array(sim.get_joint_velocities(fly.name))
    ang = (sign * (qa[q_slot] - q0[q_slot])).reshape(6, 6); om = (sign * qv[q_slot]).reshape(6, 6)
    cf, frc = sim.get_ground_contact_info(fly.name)[:2]
    contact = np.asarray(cf)[leg_perm] > 0.5; fn = np.linalg.norm(np.asarray(frc)[leg_perm], axis=1)
    if args.proprio:
        lx = leg_load()
        rates = legs.afferents(ang, om, lx)
    cord.step(drive_on=t >= DN_ON, clamp_idx=legs.s_idx if args.proprio else None, clamp_val=rates)
    tq = legs.torques(DT_N)
    lift = (legs.A[:, 0] - legs.A[:, 1]).reshape(6, 6)[:, 2] * lift_sign
    if args.adhesion == "auto":
        adh = (contact & (lift <= 0.0)).astype(float)
    else:
        adh[:] = 1.0 if args.adhesion == "on" else 0.0
    sim.set_actuator_inputs(fly.name, ActuatorType.MOTOR, sign * tq)
    sim.set_leg_adhesion_states(fly.name, adh[np.argsort(leg_perm)])
    for _ in range(int(round(DT_N / DT_P))):
        sim.step()
    if s % 25 == 0:      # 5 ms
        th = data.xpos[thor]
        row = {"t": round(t, 4), "x": th[0], "y": th[1], "z": th[2]}
        for i, leg in enumerate(LEGS):
            row[f"{leg}_contact"] = int(contact[i]); row[f"{leg}_fn"] = fn[i]; row[f"{leg}_adh"] = adh[i]
            row[f"{leg}_load"] = lx[i] if args.proprio else 0.0
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
print(f"\n{args.tag or 'run'}: amps {args.amps or '-'} proprio {args.proprio} sensors {args.sensors} adhesion {args.adhesion}")
if args.proprio:
    print("mean afferent rate by kind at the end (Hz):",
          {k: round(float(rates[legs.s_kind == i].mean()), 1) for i, k in enumerate(KINDS) if (legs.s_kind == i).any()})
print(f"displacement over the last {dur:.2f} s: forward {disp @ fwd[:2]:+.3f} mm, total {np.linalg.norm(disp):.3f} mm "
      f"({np.linalg.norm(disp) / dur:.2f} mm/s); thorax height {np.mean([r['z'] for r, k in zip(rows, sel) if k]):.3f} mm")
for leg in LEGS:
    c = np.array([r[f"{leg}_contact"] for r in rows])[sel]; td = int(np.sum((c[1:] == 1) & (c[:-1] == 0)))
    sw = np.array([r[f"{leg}_swing_mn"] for r in rows])[sel]; st = np.array([r[f"{leg}_stance_mn"] for r in rows])[sel]
    print(f"  {leg}: stance fraction {c.mean():.2f}, touchdowns {td} ({td / dur:.1f}/s), "
          f"ThC_pro range {np.ptp([r[f'{leg}_ThC_pro'] for r, k in zip(rows, sel) if k]):.2f} rad, "
          f"CTr range {np.ptp([r[f'{leg}_CTr'] for r, k in zip(rows, sel) if k]):.2f} rad, swing/stance MN {sw.mean():.0f}/{st.mean():.0f} Hz")
print("saved", out.name, f"| wall {time.time() - t0:.0f} s")
