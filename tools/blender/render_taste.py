"""Render a close-up of the labellum with taste sensors as coloured beads (does not save the .blend).
blender -b blender/fly_body.blend --python tools/blender/render_taste.py"""
import bpy, math
from mathutils import Vector
ROOT = r"F:\Fruit Fly Experiment"
col = {"L": (0.95, 0.75, 0.1, 1), "I": (0.2, 0.8, 0.3, 1), "S": (0.2, 0.5, 1, 1), "p": (0.9, 0.2, 0.8, 1), "x": (1, 1, 1, 1)}
mats = {}
for k, c in col.items():
    m = bpy.data.materials.new("bead_" + k); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = c; b.inputs["Emission Color"].default_value = c
    b.inputs["Emission Strength"].default_value = 0.6; mats[k] = m
for o in list(bpy.data.objects):
    if o.name.startswith(("lab_", "peg_")):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.006, segments=10, ring_count=6, location=o.matrix_world.translation)
        s = bpy.context.active_object
        s.data.materials.append(mats[o.name[6] if o.name.startswith("lab_") else "p"])
for o in bpy.data.objects:
    if o.name.startswith(("ommatidia", "optical_axes")):
        o.hide_render = True
target = bpy.data.objects["c_haustellum"].matrix_world @ Vector((0.25, 0, -0.1))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); bpy.context.scene.collection.objects.link(cam)
cam.location = target + Vector((0.55, -0.35, -0.35)); cam.data.lens = 50; cam.data.clip_start = 0.01
d = target - cam.location; cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
sc = bpy.context.scene; sc.camera = cam
for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
    try:
        sc.render.engine = eng; break
    except TypeError:
        pass
light = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN")); sc.collection.objects.link(light)
light.rotation_euler = (math.radians(40), math.radians(-30), 0); light.data.energy = 3
sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.05, 0.05, 0.07)
sc.render.resolution_x, sc.render.resolution_y = 1200, 800
sc.render.filepath = ROOT + r"\screenshots\phase3_taste_sensilla.png"
bpy.ops.render.render(write_still=True)
print("RENDERED", sc.render.engine)
