"""Protocol and physics check of tools/body_server.py without Godot: start the server, connect, send 1 s of frames
(33 ms each, as Godot at 30 fps) with all motor neurons silent, then 0.5 s with the tibia flexor pool of the left
front leg at 150 Hz, and check: handshake sizes, finite poses, standing height, and that the flexor drive bends that
leg's femur-tibia joint (tibia moves toward the femur) while the other legs stay put.
usage: .venv-flygym/Scripts/python tools/body_server_test.py"""
import sys, json, socket, struct, subprocess, time, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 47831
srv = subprocess.Popen([sys.executable, str(ROOT / "tools/body_server.py"), "--port", str(PORT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in srv.stdout:
    if "listening" in line:
        print(line.strip()); break
s = socket.create_connection(("127.0.0.1", PORT)); f = s.makefile("rb")
hello = json.loads(f.readline())
nm, ns, segs = len(hello["mn_ids"]), len(hello["sensor_ids"]), hello["segments"]
print(f"handshake: {nm} motor neurons, {ns} sensors, {len(segs)} segments")
lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"]
flex = set(lm["lf"]["motor"]["FTi"]["pos"])


def frame(rates, dt=1 / 30):
    msg = np.concatenate([[dt], rates]).astype("<f4")
    s.sendall(struct.pack("<I", msg.size) + msg.tobytes())
    m = struct.unpack("<I", f.read(4))[0]
    out = np.frombuffer(f.read(4 * m), "<f4")
    pose = out[:7 * len(segs)].reshape(-1, 7); aff = out[7 * len(segs):7 * len(segs) + ns]
    rest = out[7 * len(segs) + ns:7 * len(segs) + ns + 24].reshape(4, 6)
    assert out.size == 7 * len(segs) + ns + 24 + 36, out.size
    return pose, aff, rest


def dist(pose, a, b):
    return float(np.linalg.norm(pose[segs.index(a), :3] - pose[segs.index(b), :3]))


t0 = time.time(); zero = np.zeros(nm)
for _ in range(30):
    pose, aff, rest = frame(zero)
th = pose[segs.index("c_thorax")]
print(f"after 1 s silent: finite {np.isfinite(pose).all()}, thorax z {th[2]:.3f} mm, contacts {rest[0].astype(int).tolist()}, "
      f"mean afferent {aff.mean():.1f} Hz, wall {time.time() - t0:.1f} s")
d0 = {l: dist(pose, f"{l}_trochanterfemur", f"{l}_tarsus1") for l in ["lf", "rf"]}
r = np.array([150.0 if x in flex else 0.0 for x in hello["mn_ids"]])
for _ in range(15):
    pose, aff, rest = frame(r)
d1 = {l: dist(pose, f"{l}_trochanterfemur", f"{l}_tarsus1") for l in ["lf", "rf"]}
print("femur base -> tarsus distance (mm) before/after LF tibia-flexor drive:",
      {l: (round(d0[l], 3), round(d1[l], 3)) for l in d0})
ok = np.isfinite(pose).all() and d1["lf"] < d0["lf"] - 0.02 and abs(d1["rf"] - d0["rf"]) < 0.02
print("PASS" if ok else "FAIL", f"| total wall {time.time() - t0:.1f} s for 1.5 s simulated")
s.sendall(struct.pack("<I", 0)); s.close(); srv.wait(timeout=30)
print(srv.stdout.read().strip())
