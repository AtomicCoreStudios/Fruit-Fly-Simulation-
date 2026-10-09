"""FlyGym (NeuroMechFly v2, Apache-2.0) smoke tests; run with the separate environment:
    .venv-flygym/Scripts/python tools/flygym_smoke.py
1. Musculoskeletal fly (FlyMimic, Ozdil et al. 2026, bundled with FlyGym incl. meshes): activate one tibia muscle
   and report the tibia joint change (cross-check of tools/msk_muscle_test.py, which ran without meshes).
2. Free NeuroMechFly on flat ground with default contact parameters (sliding friction 1.0) and leg adhesion:
   legs held at the neutral pose by position actuators; report thorax height and leg ground contacts."""
import os, pathlib
# keep FlyGym's lazily downloaded assets inside the project (machine rule: no writes outside the project folder)
os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(pathlib.Path(__file__).resolve().parents[1] / "assets/flygym_cache"))
import numpy as np
import flygym
import flygym.anatomy as A
from flygym.compose import (ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly,
                            build_musculoskeletal_simulation)
from flygym.utils.math import Rotation3D


def test_musculoskeletal():
    sim, fly = build_musculoskeletal_simulation()
    m = sim.mj_model if hasattr(sim, "mj_model") else None
    import mujoco as mj
    model = m if m is not None else sim.physics.model if hasattr(sim, "physics") else None
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    data = sim.mj_data if hasattr(sim, "mj_data") else sim.physics.data

    def run(u):
        sim.reset()
        data.ctrl[:] = 0.0
        if u is not None:
            data.ctrl[u] = 1.0
        for _ in range(5000):
            sim.step()
        j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "joint_LFTibia_pitch")
        return float(data.qpos[model.jnt_qposadr[j]])
    base = run(None)
    for name in ("LFTibia_flex_93434", "LFTibia_extensor_93932"):
        u = names.index(name)
        print(f"musculoskeletal (with meshes): {name} -> LF tibia pitch change {run(u) - base:+.3f} rad")


def test_free_walker():
    fly = NeuroMechFly()
    skel = A.Skeleton(axis_order=A.AxisOrder.YAW_PITCH_ROLL if hasattr(A.AxisOrder, "YAW_PITCH_ROLL") else list(A.AxisOrder)[0],
                      joint_preset=A.JointPreset.LEGS_ONLY)
    fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL if hasattr(KinematicPosePreset, "NEUTRAL") else None)
    dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION) if False else None
    act = fly.add_actuators(fly.skeleton.get_actuated_dofs_from_preset(A.ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
                            if hasattr(fly.skeleton, "get_actuated_dofs_from_preset") else list(fly.jointdofs if hasattr(fly, "jointdofs") else []),
                            ActuatorType.POSITION,
                            neutral_input=KinematicPosePreset.NEUTRAL if hasattr(KinematicPosePreset, "NEUTRAL") else None,
                            kp=50.0)
    fly.add_leg_adhesion(gain=1.0)
    world = FlatGroundWorld()
    world.add_fly(fly, np.array([0.0, 0.0, 0.7]), Rotation3D("quat", [1, 0, 0, 0]), add_ground_contact_sensors=True)
    sim = flygym.Simulation(world)
    sim.reset()
    sim.set_leg_adhesion_states(fly.name, np.ones(6))
    for _ in range(int(0.5 / sim.timestep)):
        sim.step()
    pos = np.asarray(sim.get_body_positions(fly.name))
    print(f"free NeuroMechFly after 0.5 s: body segment heights (mm) min {pos[..., 2].min():.3f} / median {np.median(pos[..., 2]):.3f}; "
          f"start height 0.7")
    info = sim.get_ground_contact_info(fly.name) if hasattr(sim, "get_ground_contact_info") else None
    print("ground contact info:", type(info).__name__, (np.round(np.asarray(info[0] if isinstance(info, tuple) else info), 2).tolist()
                                                         if info is not None else None))


if __name__ == "__main__":
    print("flygym", getattr(flygym, "__version__", "2.x"), "| timestep handled per simulation")
    for t in (test_musculoskeletal, test_free_walker):
        try:
            t()
        except Exception as e:
            import traceback
            print(f"{t.__name__} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
