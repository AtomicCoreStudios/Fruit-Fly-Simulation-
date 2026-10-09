"""Live physics body for the Godot brain: one MuJoCo NeuroMechFly (tools/nmf_body.py) driven by the brain's real leg
motor neurons; returns every body segment's pose (for rendering, incl. the head that carries the compound eyes) and the
rates of the leg's real proprioceptor neurons (tools/vnc_body_model.py rules; presynaptic inhibition stays in Godot).
Started by Godot with --physics=mujoco (scripts/fly_legs.gd), or by hand. Listens on 127.0.0.1 only.

Protocol (little endian):
  on connect, server -> client: one JSON line {"version", "mn_ids", "sensor_ids", "segments", "dt_phys", "weight_uN",
                                               "legs", "rest_poses" (neutral pose, 7 per segment)}
  each frame, client -> server: uint32 n, float32[n] = [dt_s, rate_Hz per mn_ids entry]     (n = 0: quit)
  server -> client: uint32 m, float32[m] = [7 per segment (pos mm x,y,z; quat w,x,y,z; MuJoCo world, z up),
                                            rate_Hz per sensor_ids entry, stance x6, load x6, pad x6,
                                            foot height mm x6, joint angle rad x36 (leg-major, ThC_pro..TiTa,
                                            biological sign re neutral), head yaw/roll rad x2 (+ = left)]
  mn_ids / sensor_ids: leg motor neurons then neck motor neurons (n_leg_mn); leg then neck proprioceptors
Motor activation and torque are updated every 1 ms of simulated time inside a frame (MuJoCo steps of 0.1 ms).
usage: .venv-flygym/Scripts/python tools/body_server.py [--port 47830] [--pad_fmax 10] [--k_joint 10]"""
import sys, json, socket, struct, time, argparse, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from nmf_body import Body, DT_P, LEGS
from vnc_body_model import Legs, Neck

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=47830)
ap.add_argument("--adhesion", default="pad")
ap.add_argument("--pad_fmax", type=float, default=10.0)
ap.add_argument("--k_joint", type=float, default=10.0)
ap.add_argument("--memory_mb", type=float, default=64.0)
args = ap.parse_args()

legs = Legs(None, k_joint=args.k_joint)
neck = Neck(None, k_joint=args.k_joint)
N_LEG_MN, N_LEG_S = len(legs.mn_idx), len(legs.s_idx)
body = Body(k_joint=args.k_joint, adhesion=args.adhesion, pad_fmax=args.pad_fmax, memory_mb=args.memory_mb)
hello = {"version": 2, "mn_ids": [int(x) for x in legs.mn_idx] + [int(x) for x in neck.mn_ids],
         "sensor_ids": [int(x) for x in legs.s_idx] + [int(x) for x in neck.s_ids],
         "n_leg_mn": N_LEG_MN, "n_leg_sensors": N_LEG_S,
         "segments": body.seg_names, "dt_phys": DT_P, "weight_uN": body.W, "legs": LEGS,
         "rest_poses": body.segment_poses().round(6).tolist()}     # neutral pose at reset (renderer self-check)
CHUNK = int(round(0.001 / DT_P))


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionError("client closed")
        buf += part
    return bytes(buf)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port)); srv.listen(1)
print(f"body server: {len(body.seg_names)} segments, {len(hello['mn_ids'])} motor neurons, {len(hello['sensor_ids'])} "
      f"proprioceptors; listening on 127.0.0.1:{args.port}", flush=True)
conn, _ = srv.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
conn.sendall((json.dumps(hello) + "\n").encode())
frames, sim_t, wall0 = 0, 0.0, time.time()
try:
    while True:
        n = struct.unpack("<I", recv_exact(conn, 4))[0]
        if n == 0:
            break
        msg = np.frombuffer(recv_exact(conn, 4 * n), dtype="<f4")
        dt, rates = float(msg[0]), msg[1:].astype(float)
        steps = max(1, int(round(dt / DT_P)))
        done = 0
        while done < steps:
            k = min(CHUNK, steps - done)
            ang, om, contact, fn = body.state()
            tq = legs.torques(k * DT_P, rates[:N_LEG_MN])
            ntq = neck.torques(k * DT_P, rates[N_LEG_MN:])
            lift = (legs.A[:, 0] - legs.A[:, 1]).reshape(6, 6)[:, 2] * body.lift_sign
            body.step(tq, lift, contact, k, ntq)
            done += k
        ang, om, contact, fn = body.state()
        lx = body.leg_load()
        ha, hv = body.head_state()
        aff = np.concatenate([legs.afferents(ang, om, lx), neck.afferents(ha, hv)])
        foot_z = body.data.xpos[body.tip_b][:, 2]
        out = np.concatenate([body.segment_poses().ravel(), aff, (contact | (body.adh > 0)).astype(float), lx,
                              body.adh, foot_z, ang.ravel(), ha]).astype("<f4")
        conn.sendall(struct.pack("<I", out.size) + out.tobytes())
        frames += 1; sim_t += steps * DT_P
        if frames % 100 == 0:
            print(f"body server: {frames} frames, {sim_t:.2f} s simulated, {sim_t / (time.time() - wall0):.2f}x real time, "
                  f"thorax z {body.data.xpos[body.thor][2]:.3f} mm", flush=True)
except (ConnectionError, OSError) as e:
    print("body server: connection ended:", e, flush=True)
finally:
    conn.close(); srv.close()
    print("body server: stopped", flush=True)
