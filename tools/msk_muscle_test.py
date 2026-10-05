"""Validate the leg motor map against the musculoskeletal fly front-leg model (FlyGym assets,
assets/flygym/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml; Hill-type muscles converted from
OpenSim with MyoConverter; Apache-2.0). Meshes are not shipped, so mesh geoms are stripped (explicit <inertial>
elements keep the mass properties). Each muscle is fully activated alone for 0.5 s; the steady-state change of every
left-front joint relative to the passive (unactivated) state is reported -> data/tables/msk_muscle_effects.csv"""
import re, pathlib
import numpy as np, pandas as pd, mujoco
ROOT = pathlib.Path(__file__).resolve().parents[1]
x = (ROOT / "assets/flygym/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml").read_text(encoding="utf-8")
x = re.sub(r'<mesh [^>]*/>', '', x); x = re.sub(r'<geom [^>]*mesh="[^"]*"[^>]*/>', '', x)
m = mujoco.MjModel.from_xml_string(x)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
lf = [i for i, n in enumerate(jn) if n.startswith("joint_LF")]
def settle(u=None, steps=5000):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    d.ctrl[:] = 0.0
    if u is not None:
        d.ctrl[u] = 1.0
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d.qpos.copy()
base = settle()           # passive settling under gravity, no activation (subtracted below)
rows = []
for u in range(m.nu):
    dq = settle(u) - base
    rows.append({"muscle": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, u),
                 **{jn[j].replace("joint_LF", ""): round(float(dq[m.jnt_qposadr[j]]), 3) for j in lf}})
r = pd.DataFrame(rows)
out = ROOT / "data/tables/msk_muscle_effects.csv"; r.to_csv(out, index=False)
pd.set_option("display.width", 200); print(r.to_string(index=False))
