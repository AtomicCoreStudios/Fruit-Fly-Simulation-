"""Phase 1: assemble the NeuroMechFly v2 body in Blender (run inside Blender).

Source: FlyGym (Apache-2.0, NeLy-EPFL), commit in assets/flygym/SOURCE_COMMIT.txt.
Meshes come from a micro-CT scan of one adult female fly (basis: measured), simplified to
<=2000 faces per segment by FlyGym. Body frames are from rigging.yaml (data/flygym_rig.json).
Right-side meshes are the left meshes mirrored in Y, exactly as FlyGym does (basis: approximate,
real flies are not perfectly symmetric).

Units: 1 Blender unit = 1 mm. Axes follow FlyGym: +X anterior, +Y left, +Z dorsal.
Each body is an Empty named after the segment (e.g. "lf_tibia") whose local transform is the
MuJoCo body frame; the mesh object "<segment>_mesh" is its child. Joint origins are the Empty
origins, so a joint is posed by rotating that Empty.
"""
import bpy, json, struct, pathlib, mathutils

ROOT = pathlib.Path(r"F:\Fruit Fly Experiment")
MESH_DIR = ROOT / "assets/flygym/model/neuromechfly/meshes/simplified_max2000faces"
RIG = json.loads((ROOT / "data/flygym_rig.json").read_text())
SCALE = 1000.0  # STL is in metres -> mm

# Colours from FlyGym visuals.yaml (RGBA)
COLOURS = {
    "eye": (0.67, 0.21, 0.12, 1.0),
    "wing": (0.8, 0.8, 0.9, 0.3),
    "arista": (0.26, 0.2, 0.16, 1.0),
    "haltere": (0.59, 0.43, 0.24, 1.0),
    "cuticle": (0.59, 0.39, 0.12, 1.0),
    "leg": (0.59, 0.39, 0.12, 1.0),
}
# Abdomen/legs are not in visuals.yaml with a specific colour here; FlyGym uses the
# same brown cuticle. Abdomen darker bands are not modelled (approximate).


def read_stl(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * n != len(data):
        raise ValueError(f"{path.name}: not a binary STL")
    verts, faces, index = [], [], {}
    for i in range(n):
        off = 84 + 50 * i + 12
        tri = []
        for k in range(3):
            v = struct.unpack_from("<3f", data, off + 12 * k)
            key = tuple(round(c, 9) for c in v)
            if key not in index:
                index[key] = len(verts)
                verts.append(v)
            tri.append(index[key])
        if len(set(tri)) == 3:
            faces.append(tri)
    return verts, faces


def material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.45
    if rgba[3] < 1:
        bsdf.inputs["Alpha"].default_value = rgba[3]
        for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            try:
                setattr(mat, attr, val)
            except (AttributeError, TypeError):
                pass
    mat.diffuse_color = rgba
    return mat


def colour_key(seg):
    link = seg.split("_", 1)[1]
    if link in COLOURS:
        return link
    if seg[:2] in ("lf", "lm", "lh", "rf", "rm", "rh"):
        return "leg"
    return "cuticle"


def build():
    scene = bpy.data.scenes.get("FlyBody") or bpy.data.scenes.new("FlyBody")
    bpy.context.window.scene = scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    coll = bpy.data.collections.get("fly_body")
    if coll is None:
        coll = bpy.data.collections.new("fly_body")
        scene.collection.children.link(coll)
    for ob in list(coll.objects):  # rebuild idempotently (only our own collection)
        bpy.data.objects.remove(ob, do_unlink=True)

    empties = {}
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        p = RIG[n]["parent"]
        if p:
            visit(p)
        seen.add(n); order.append(n)
    for n in RIG:
        visit(n)

    for seg in order:
        info = RIG[seg]
        e = bpy.data.objects.new(seg, None)
        e.empty_display_type = "ARROWS"
        e.empty_display_size = 0.08
        coll.objects.link(e)
        if info["parent"]:
            e.parent = empties[info["parent"]]
        w, x, y, z = info["quat"]
        e.matrix_parent_inverse.identity()
        e.matrix_basis = mathutils.Matrix.Translation(info["pos"]) @ \
            mathutils.Quaternion((w, x, y, z)).normalized().to_matrix().to_4x4()
        e["mass_kg"] = info["mass"]
        e["basis"] = "measured (FlyGym rigging, micro-CT)"
        empties[seg] = e

        mirrored = seg.startswith("r")
        src = ("l" + seg[1:]) if mirrored else seg
        verts, faces = read_stl(MESH_DIR / f"{src}.stl")
        sy = -SCALE if mirrored else SCALE
        verts = [(vx * SCALE, vy * sy, vz * SCALE) for vx, vy, vz in verts]
        if mirrored:
            faces = [f[::-1] for f in faces]  # keep normals outward after reflection
        me = bpy.data.meshes.new(f"{seg}_mesh")
        me.from_pydata(verts, [], faces)
        me.validate(); me.update()
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(f"{seg}_mesh", me)
        coll.objects.link(ob)
        ob.parent = e
        ob.data.materials.append(material(f"fly_{colour_key(seg)}", COLOURS[colour_key(seg)]))
        ob["source_mesh"] = f"{src}.stl" + (" (mirrored Y)" if mirrored else "")
        ob["basis"] = "approximate (mirror of left)" if mirrored else "measured (micro-CT, simplified)"
    return scene, empties


AXIS = {"pitch": (0, 1, 0), "roll": (0, 0, 1), "yaw": (1, 0, 0)}  # flygym/anatomy.py


def rest_matrix(seg):
    w, x, y, z = RIG[seg]["quat"]
    return mathutils.Matrix.Translation(RIG[seg]["pos"]) @ \
        mathutils.Quaternion((w, x, y, z)).normalized().to_matrix().to_4x4()


def apply_pose(pose_file):
    """Apply a FlyGym KinematicPose yaml (hinge DoFs chained in axis_order, MuJoCo style).
    Left angles are copied to the right side, and right-side roll/yaw axes are negated,
    exactly as FlyGym does (base_fly.py, pose.py)."""
    lines = pathlib.Path(pose_file).read_text().splitlines()
    unit_deg = any("degree" in l for l in lines if l.startswith("angle_unit"))
    order = []
    angles = {}
    in_order = False
    for l in lines:
        s = l.strip()
        if s.startswith("axis_order:"):
            rest = s.split(":", 1)[1].strip()
            if rest.startswith("["):
                order = [a.strip() for a in rest.strip("[]").split(",")]
            else:
                in_order = True
            continue
        if in_order and s.startswith("- "):
            order.append(s[2:].strip()); continue
        in_order = False
        if s.count("-") >= 2 and ":" in s and not s.startswith("#"):
            k, v = s.split(":"); angles[k.strip()] = float(v)
    for k, v in list(angles.items()):
        p, c, a = k.split("-")
        if c.startswith("l"):
            rk = f"{'r' + p[1:] if p.startswith('l') else p}-r{c[1:]}-{a}"
            angles.setdefault(rk, v)
    import math
    per_seg = {}
    for k, v in angles.items():
        p, c, a = k.split("-")
        per_seg.setdefault(c, {})[a] = math.radians(v) if unit_deg else v
    for seg in RIG:
        R = mathutils.Matrix.Identity(4)
        for a in order:
            ang = per_seg.get(seg, {}).get(a, 0.0)
            vec = mathutils.Vector(AXIS[a])
            if seg.startswith("r") and a != "pitch":
                vec = -vec
            R = R @ mathutils.Matrix.Rotation(ang, 4, vec)
        bpy.data.objects[seg].matrix_basis = rest_matrix(seg) @ R
        bpy.data.objects[seg]["pose"] = pathlib.Path(pose_file).name


scene, empties = build()
apply_pose(ROOT / "assets/flygym/model/neuromechfly/pose/neutral/pitch_roll_yaw.yaml")
for area in bpy.context.window.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.clip_start = 0.005
        area.spaces.active.clip_end = 1000
print("built", len(empties), "segments")
