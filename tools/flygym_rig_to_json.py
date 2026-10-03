"""Convert FlyGym's rigging.yaml (body frames, measured from micro-CT; NeuroMechFly v2)
plus its parent tree (flygym/anatomy.py ALL_CONNECTED_SEGMENT_PAIRS) into data/flygym_rig.json.
Units: mm. quat is MuJoCo order (w, x, y, z), relative to the parent body."""
import json, re, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
src = root / "assets/flygym/model/neuromechfly/rigging.yaml"
rig, cur = {}, None
for line in src.read_text().splitlines():
    if re.match(r"^\w+:\s*$", line):
        cur = line.strip()[:-1]; rig[cur] = {}
    elif cur and ":" in line:
        k, v = line.strip().split(":", 1); v = v.strip()
        rig[cur][k] = json.loads(v) if v.startswith("[") else float(v)
SIDES, LEGS = "lr", [s + p for s in "lr" for p in "fmh"]
LEG = ["coxa", "trochanterfemur", "tibia"] + [f"tarsus{i}" for i in range(1, 6)]
def chain(*a): return [(a[i], a[i + 1]) for i in range(len(a) - 1)]
pairs = [("c_thorax", "c_head")] + chain("c_head", "c_rostrum", "c_haustellum") \
    + chain("c_thorax", "c_abdomen12", *[f"c_abdomen{i}" for i in "3456"])
for s in SIDES:
    pairs += [("c_head", f"{s}_eye")] + chain("c_head", f"{s}_pedicel", f"{s}_funiculus", f"{s}_arista")
    pairs += [("c_thorax", f"{s}_wing"), ("c_thorax", f"{s}_haltere")]
for l in LEGS:
    pairs += chain("c_thorax", *[f"{l}_{k}" for k in LEG])
parent = {c: p for p, c in pairs}
out = {n: {**rig[n], "parent": parent.get(n)} for n in rig}
(root / "data/flygym_rig.json").write_text(json.dumps(out, indent=1))
print(len(out), "segments;", sum(1 for v in out.values() if v["parent"] is None), "root")
