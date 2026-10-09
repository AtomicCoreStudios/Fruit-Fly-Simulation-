"""The physical fly body: NeuroMechFly in MuJoCo (FlyGym 2.1.0), true scale (mm, g, s; 1.024 mg, weight 10.05 uN),
driven by joint torques in 'biological' sign and read out as joint state, cuticular leg load and ground contact.
Shared by tools/nmf_closed_loop.py (offline, with the nerve-cord rate model) and tools/body_server.py (live bridge to
the Godot brain). Run with .venv-flygym.

Joints: full biological skeleton (126 DoFs incl. head/neck, so the head that carries the eyes is a physical body);
passive springs K (springref = NeuroMechFly neutral pose), damping K x 30 ms. The 36 leg DoFs of our motor map
(ThC_pro, ThC_add, CTr, FeRot, FTi, TiTa per leg) get motor actuators; all other joints are passive.
Adhesion modes: 'pad' (default) = per-leg switchable connect constraint to the touchdown point that holds only what the
leg loads it with, breaks above pad_fmax or when the leg actively levates (peeling); 'auto'/'on'/'off' = FlyGym's
adhesion actuator (constant pull while in contact; not physical, kept for comparison).
Load: bending moment carried through each femur base (MuJoCo cfrc_int), perpendicular to the femur, in units of
M_REF = (W/3) x leg length."""
import os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
# keep FlyGym's lazily downloaded assets inside the project (machine rule: no writes outside the project folder)
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(ROOT / "assets/flygym_cache"))
import numpy as np
import mujoco as mj
import flygym
import flygym.anatomy as A
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
DOFS = ["ThC_pro", "ThC_add", "CTr", "FeRot", "FTi", "TiTa"]
MJ_DOF = {"ThC_pro": "c_thorax-{l}_coxa-pitch", "ThC_add": "c_thorax-{l}_coxa-roll",
          "CTr": "{l}_coxa-{l}_trochanterfemur-pitch", "FeRot": "{l}_coxa-{l}_trochanterfemur-roll",
          "FTi": "{l}_trochanterfemur-{l}_tibia-pitch", "TiTa": "{l}_tibia-{l}_tarsus1-pitch"}
DT_P = 0.0001                     # MuJoCo step (FlyGym default)
# neck: FlyGym names its head joint axes in its own convention (the "roll" joint turns about the vertical axis);
# mapped here to biological head yaw/roll and checked kinematically in Body._signs
NECK_DOF = {"yaw": "c_thorax-c_head-roll", "roll": "c_thorax-c_head-yaw"}


class Body:
    def __init__(self, k_joint=10.0, skeleton="ALL_BIOLOGICAL", adhesion="pad", pad_fmax=10.0, adh_gain=40.0,
                 spawn_z=0.7, memory_mb=None):
        self.adhesion, self.pad_fmax = adhesion, pad_fmax
        self.k = k_joint
        fly = NeuroMechFly()
        skel = A.Skeleton(axis_order=A.AxisOrder.PITCH_ROLL_YAW, joint_preset=A.JointPreset[skeleton])
        fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL, stiffness=k_joint, damping=k_joint * 0.030)
        jorder = fly.get_jointdofs_order(); self.jname = [j.name for j in jorder]
        act_dofs = [jorder[self.jname.index(MJ_DOF[d].format(l=leg))] for leg in LEGS for d in DOFS]
        fly.add_actuators(act_dofs, ActuatorType.MOTOR, forcerange=(-30.0, 30.0))
        fly.add_actuators([jorder[self.jname.index(NECK_DOF[a])] for a in ("yaw", "roll")], ActuatorType.MOTOR,
                          forcerange=(-30.0, 30.0))
        fly.add_leg_adhesion(gain=adh_gain if adhesion != "pad" else 0.0)
        world = FlatGroundWorld()
        world.add_fly(fly, np.array([0.0, 0.0, spawn_z]), Rotation3D("quat", [1, 0, 0, 0]), add_ground_contact_sensors=True)
        if memory_mb:
            world.mjcf_root.memory = int(memory_mb * 1024 * 1024)
        pad_names = []
        for leg in LEGS:
            e = world.mjcf_root.add_equality(type=mj.mjtEq.mjEQ_CONNECT, objtype=mj.mjtObj.mjOBJ_BODY,
                                             name1=f"{fly.name}/{leg}_tarsus5", name2="world", active=False)
            e.name = f"pad_{leg}"; e.solref = [0.002, 1.0]
            pad_names.append(e.name)
        self.fly, self.sim = fly, flygym.Simulation(world, timestep=DT_P)
        self.sim.reset()
        m, d = self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.pad_id = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, n) for n in pad_names]
        d.eq_active[:] = 0
        self.pad_on = np.zeros(6, bool); self.pad_ready = np.ones(6, bool); self.pad_force = np.zeros(6)
        self.adh = np.zeros(6)
        legs_order = list(fly.get_legs_order()); self.leg_perm = [legs_order.index(l) for l in LEGS]
        self.q_slot = np.array([self.jname.index(MJ_DOF[dd].format(l=leg)) for leg in LEGS for dd in DOFS])
        self.q0 = np.array(self.sim.get_joint_angles(fly.name)).copy()
        self.n_slot = np.array([self.jname.index(NECK_DOF[a]) for a in ("yaw", "roll")])
        order = [j.name for j in fly.get_actuated_jointdofs_order(ActuatorType.MOTOR)]
        assert order[:36] == [self.jname[i] for i in self.q_slot] and order[36:] == [NECK_DOF["yaw"], NECK_DOF["roll"]], order
        # all segment bodies (names without the 'nmf/' prefix), for rendering
        self.seg_names, self.seg_ids = [], []
        for i in range(m.nbody):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or ""
            if nm.startswith(fly.name + "/"):
                self.seg_names.append(nm[len(fly.name) + 1:]); self.seg_ids.append(i)
        self.thor = self._bid("c_thorax")
        mj.mj_kinematics(m, d)
        fwd = d.xmat[self.thor].reshape(3, 3)[:, 0].copy(); fwd[2] = 0; self.fwd = fwd / np.linalg.norm(fwd)
        self._signs()
        self.W = float(m.body_subtreemass[self.thor]) * 9810.0
        self.fem_b = [self._bid(f"{l}_trochanterfemur") for l in LEGS]; self.tib_b = [self._bid(f"{l}_tibia") for l in LEGS]
        self.tip_b = [self._bid(f"{l}_tarsus5") for l in LEGS]
        self.M_REF = np.array([self.W / 3 * np.linalg.norm(d.xpos[t] - d.xpos[f]) for f, t in zip(self.fem_b, self.tip_b)])

    def _bid(self, part):
        return self.seg_ids[self.seg_names.index(part)]

    def _qadr(self, name):
        for j in range(self.model.njnt):
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith(name):
                return self.model.jnt_qposadr[j]
        raise KeyError(name)

    def _signs(self):
        """+ must protract, adduct, flex (CTr, FTi), lower the tarsus (TiTa), as in fly_legs.gd _fix_sign."""
        m, d = self.model, self.data
        up = np.array([0, 0, 1.0]); self.sign = np.ones(36); self.lift_sign = np.ones(6); qs = d.qpos.copy()
        for li, leg in enumerate(LEGS):
            tip, cox, fem = self._bid(f"{leg}_tarsus5"), self._bid(f"{leg}_coxa"), self._bid(f"{leg}_trochanterfemur")
            mj.mj_kinematics(m, d); p0 = d.xpos[tip].copy(); c0 = d.xpos[cox].copy(); f0 = d.xpos[fem].copy()
            lat = c0 - d.xpos[self.thor]; lat[2] = 0; lat /= np.linalg.norm(lat)
            for di, dd in enumerate(DOFS):
                d.qpos[:] = qs; d.qpos[self._qadr(MJ_DOF[dd].format(l=leg))] += 0.05; mj.mj_kinematics(m, d)
                dp = d.xpos[tip] - p0
                metric = {"ThC_pro": dp @ self.fwd, "ThC_add": -(dp @ lat) - (dp @ up),
                          "CTr": np.linalg.norm(p0 - c0) - np.linalg.norm(d.xpos[tip] - c0),
                          "FTi": np.linalg.norm(p0 - f0) - np.linalg.norm(d.xpos[tip] - f0),
                          "TiTa": -(dp @ up), "FeRot": 1.0}[dd]
                self.sign[li * 6 + di] = 1.0 if metric >= 0 else -1.0
                if dd == "CTr":
                    self.lift_sign[li] = 1.0 if (dp @ up) * self.sign[li * 6 + di] > 0 else -1.0
        # head: + yaw = head turns to the left (its anterior axis gains +y); + roll = left side down
        h = self._bid("c_head"); self.neck_sign = np.ones(2)
        mj.mj_kinematics(m, d); R0 = d.xmat[h].reshape(3, 3).copy()
        for k, a in enumerate(("yaw", "roll")):
            d.qpos[:] = qs; d.qpos[self._qadr(NECK_DOF[a])] += 0.05; mj.mj_kinematics(m, d)
            R1 = d.xmat[h].reshape(3, 3)
            metric = (R1[1, 0] - R0[1, 0]) if a == "yaw" else -(R1[2, 1] - R0[2, 1])
            assert abs(metric) > 1e-3, f"neck axis mapping wrong for {a}"
            self.neck_sign[k] = 1.0 if metric > 0 else -1.0
        d.qpos[:] = qs; mj.mj_forward(m, d)

    def head_state(self):
        """(angle, velocity) of the head re neutral, (yaw, roll), biological sign."""
        qa = np.array(self.sim.get_joint_angles(self.fly.name)); qv = np.array(self.sim.get_joint_velocities(self.fly.name))
        return self.neck_sign * (qa[self.n_slot] - self.q0[self.n_slot]), self.neck_sign * qv[self.n_slot]

    def state(self):
        """(ang, om) as (6 legs, 6 dofs) in biological sign re neutral; contact flags (6,); contact force norms (6,)."""
        qa = np.array(self.sim.get_joint_angles(self.fly.name)); qv = np.array(self.sim.get_joint_velocities(self.fly.name))
        ang = (self.sign * (qa[self.q_slot] - self.q0[self.q_slot])).reshape(6, 6)
        om = (self.sign * qv[self.q_slot]).reshape(6, 6)
        cf, frc = self.sim.get_ground_contact_info(self.fly.name)[:2]
        return ang, om, np.asarray(cf)[self.leg_perm] > 0.5, np.linalg.norm(np.asarray(frc)[self.leg_perm], axis=1)

    def leg_load(self):
        m, d = self.model, self.data
        mj.mj_rnePostConstraint(m, d)
        out = np.zeros(6)
        for i, (b, tb) in enumerate(zip(self.fem_b, self.tib_b)):
            c = d.subtree_com[m.body_rootid[b]]; tq, f = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            tp = tq + np.cross(c - d.xpos[b], f)
            ax = d.xpos[tb] - d.xpos[b]; ax /= np.linalg.norm(ax)
            out[i] = np.linalg.norm(tp - (tp @ ax) * ax) / self.M_REF[i]
        return out

    def step(self, torque_bio, lift, contact, n_sub, neck_torque=None):
        """Apply leg torques (36, biological sign) and neck torques ((yaw, roll), biological sign), update adhesion,
        advance n_sub MuJoCo steps. lift: (6,) net trochanter drive in the lifting direction (> 0 = levating)."""
        m, d = self.model, self.data
        nt = np.zeros(2) if neck_torque is None else np.asarray(neck_torque, float)
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.MOTOR,
                                     np.concatenate([self.sign * torque_bio, self.neck_sign * nt]))
        if self.adhesion == "pad":
            self.pad_force[:] = 0.0
            for k in np.where(d.efc_type[:d.nefc] == mj.mjtConstraint.mjCNSTR_EQUALITY)[0]:
                j = int(d.efc_id[k])
                if j in self.pad_id:
                    self.pad_force[self.pad_id.index(j)] += d.efc_force[k] ** 2
            self.pad_force[:] = np.sqrt(self.pad_force)
            for i in range(6):
                if self.pad_on[i] and (self.pad_force[i] > self.pad_fmax or lift[i] > 0.0):
                    self.pad_on[i] = False; d.eq_active[self.pad_id[i]] = 0; self.pad_ready[i] = False
                elif not self.pad_on[i] and contact[i] and lift[i] <= 0.0 and self.pad_ready[i]:
                    e = self.pad_id[i]
                    m.eq_data[e, :3] = 0.0; m.eq_data[e, 3:6] = d.xpos[self.tip_b[i]]
                    d.eq_active[e] = 1; self.pad_on[i] = True
                if not contact[i]:
                    self.pad_ready[i] = True
            self.adh = self.pad_on.astype(float)
        else:
            self.adh = ((contact & (lift <= 0.0)) if self.adhesion == "auto" else
                        np.full(6, self.adhesion == "on")).astype(float)
            self.sim.set_leg_adhesion_states(self.fly.name, self.adh[np.argsort(self.leg_perm)])
        for _ in range(n_sub):
            self.sim.step()

    def segment_poses(self):
        """(n_seg, 7): world position (mm) and quaternion (w, x, y, z) of every segment body."""
        d = self.data
        return np.concatenate([d.xpos[self.seg_ids], d.xquat[self.seg_ids]], axis=1)
