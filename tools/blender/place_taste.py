"""Phase 3 (Blender, headless OK): place taste sensor empties on the NeuroMechFly body and re-export
assets/fly/fly.glb. Run:  blender -b blender/fly_body.blend --python tools/blender/place_taste.py

Sensors (names are the keys of data/taste_organs.json, built by tools/build_taste.py):
  lab_<l|r>_<L|I|S>NN  labellar taste bristles on the outer surface of each labellar lobe
                       (distal haustellum mesh). L-type outermost, then I, then S. Real bristle
                       positions are not measured in this fly: basis approximate.
  peg_<l|r>_NN         taste pegs on the inner (oral) labellar surface: basis approximate.
  leg_<leg>_tarsusK    one contact sensor on the ventral surface of each tarsal segment.
  pharynx              one internal sensor inside the rostrum (ingested food).
"""
import bpy, json, pathlib
import numpy as np
from mathutils import Vector

ROOT = pathlib.Path(r"F:\Fruit Fly Experiment")
ORG = json.loads((ROOT / "data/taste_organs.json").read_text())
coll = bpy.data.collections["fly_body"]
for ob in list(coll.objects):
    if ob.name.startswith(("lab_", "peg_", "leg_", "pharynx")):
        bpy.data.objects.remove(ob, do_unlink=True)


def fps(P, k, seed=0):
    """farthest-point sampling of k points from P (n,3)"""
    if len(P) <= k:
        return list(range(len(P)))
    idx = [int(np.argmax(P[:, 0]))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return idx


def add_empty(name, parent, co, kind):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "SPHERE"; e.empty_display_size = 0.008
    coll.objects.link(e); e.parent = bpy.data.objects[parent]
    e.matrix_parent_inverse.identity(); e.location = Vector(co)
    e["sensor_kind"] = kind
    return e


positions = {}
# ---- labellum: distal 35% of the haustellum mesh, local frame of c_haustellum
mh = bpy.data.objects["c_haustellum_mesh"]


def surface_samples(me, n=20000, seed=7):
    """area-weighted random points on the mesh surface, with face normals (the simplified mesh has
    too few vertices on the labellum to host ~65 sensilla per side)"""
    me.calc_loop_triangles()
    T = np.array([[me.vertices[i].co[:] for i in t.vertices] for t in me.loop_triangles])
    Nf = np.array([t.normal[:] for t in me.loop_triangles])
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    f = rng.choice(len(T), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n); s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * T[f, 0] + (s1 * (1 - r2))[:, None] * T[f, 1] + (s1 * r2)[:, None] * T[f, 2]
    return P, Nf[f]


V, N = surface_samples(mh.data)
xmax = V[:, 0].max()
distal = V[:, 0] > xmax - 0.12
for side, sgn in (("left", 1), ("right", -1)):
    sens = [s for s in ORG["sensilla"] if s["side"] == side]
    bris = [s for s in sens if s["class"] in "LIS"]
    pegs = [s for s in sens if s["class"] == "peg"]
    lobe = distal & (sgn * V[:, 1] > 0.02)
    outer = lobe & (sgn * N[:, 1] > 0.2)            # facing outward (lateral)
    inner = lobe & (N[:, 2] < -0.3) & (sgn * V[:, 1] < 0.12)   # oral surface, facing down/medially
    Po = V[outer]; pick = fps(Po, len(bris))
    order = sorted(pick, key=lambda i: -sgn * Po[i, 1])        # most lateral first -> L, I, S
    cls_sorted = sorted(bris, key=lambda s: "LIS".index(s["class"]))
    for s, i in zip(cls_sorted, order):
        add_empty(s["id"], "c_haustellum", Po[i] + 0.004 * N[outer][i], "labellum_bristle")
        positions[s["id"]] = {"parent": "c_haustellum", "pos": Po[i].round(4).tolist()}
    Pi = V[inner]; pick = fps(Pi, len(pegs))
    for s, i in zip(pegs, pick):
        add_empty(s["id"], "c_haustellum", Pi[i], "labellum_peg")
        positions[s["id"]] = {"parent": "c_haustellum", "pos": Pi[i].round(4).tolist()}
# ---- legs: ventral contact point of each tarsal segment
for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]:
    for k in range(1, 6):
        seg = f"{leg}_tarsus{k}"
        m = bpy.data.objects[f"{seg}_mesh"]
        W = np.array([(m.matrix_world @ v.co)[:] for v in m.data.vertices])
        L = np.array([v.co[:] for v in m.data.vertices])
        i = int(np.argmin(W[:, 2]))                     # lowest point in the standing pose
        name = f"leg_{leg}_tarsus{k}"
        add_empty(name, seg, L[i], "leg")
        positions[name] = {"parent": seg, "pos": L[i].round(4).tolist()}
# ---- pharynx: centroid of the rostrum mesh
mr = bpy.data.objects["c_rostrum_mesh"]
R = np.array([v.co[:] for v in mr.data.vertices]).mean(0)
add_empty("pharynx", "c_rostrum", R, "pharynx")
positions["pharynx"] = {"parent": "c_rostrum", "pos": R.round(4).tolist()}

ORG["sensor_positions"] = positions
(ROOT / "data/taste_organs.json").write_text(json.dumps(ORG, indent=1))
bpy.ops.wm.save_mainfile(filepath=str(ROOT / "blender/fly_body.blend"))
for o in bpy.context.view_layer.objects:
    o.select_set(False)
for o in coll.objects:
    if not o.name.startswith("optical_axes"):
        o.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(ROOT / "assets/fly/fly.glb"), export_format="GLB",
                          use_selection=True, use_active_scene=True, export_extras=True, export_yup=True)
print(f"TASTE: placed {len(positions)} sensors and re-exported fly.glb")
