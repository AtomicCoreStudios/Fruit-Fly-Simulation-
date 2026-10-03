"""Phase 2 (Blender): put the 1709 measured ommatidia (data/ommatidia.json) onto the
NeuroMechFly eye meshes.

The micro-CT eye (Zhao et al. 2025) and the NeuroMechFly body are different flies, so the lens
cloud is fitted to the two eye meshes with a similarity transform (scale, rotation, translation;
iterative closest point), then each lens is snapped to the nearest point of its eye mesh.
Facet POSITIONS on the body are therefore approximate (fit residual is reported). Optical AXES
stay measured; they are only rotated by the same rigid fit so they stay attached to the head.

Output:
  - objects "ommatidia_right" / "ommatidia_left" (child of c_head): one small hexagonal facet
    cap per ommatidium, coloured yellow / pale / DRA, with per-face attribute "facet_id"
  - objects "optical_axes_right/left": line per facet along its optical axis (hidden in render)
  - data/ommatidia_placement.json: per facet, position and axis in the c_head frame (mm)
"""
import bpy, bmesh, json, math, pathlib
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ROOT = pathlib.Path(r"F:\Fruit Fly Experiment")
OM = json.loads((ROOT / "data/ommatidia.json").read_text())
FACETS = OM["facets"]
head = bpy.data.objects["c_head"]
bpy.context.view_layer.update()
H_inv = head.matrix_world.inverted()


def eye_mesh_head(side):
    ob = bpy.data.objects[f"{'r' if side == 'right' else 'l'}_eye_mesh"]
    M = H_inv @ ob.matrix_world
    verts = [M @ v.co for v in ob.data.vertices]
    polys = [tuple(p.vertices) for p in ob.data.polygons]
    return np.array(verts), BVHTree.FromPolygons(verts, polys)


def umeyama(src, dst):
    """Similarity transform (s, R, t) minimising |s R src + t - dst|."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def fit():
    V = {s: eye_mesh_head(s) for s in ("right", "left")}
    P = np.array([f["lens_um"] for f in FACETS]) / 1000.0     # um -> mm
    side = np.array([f["side"] for f in FACETS])
    # Initial guess: identity rotation (both frames are anterior/left/dorsal), match centroids + spread.
    both = np.vstack([V["right"][0], V["left"][0]])
    s = np.sqrt(((both - both.mean(0)) ** 2).sum(1).mean() / ((P - P.mean(0)) ** 2).sum(1).mean())
    R, t = np.eye(3), both.mean(0) - s * P.mean(0)
    for it in range(60):
        Q = (s * (R @ P.T)).T + t
        tgt = np.empty_like(Q)
        for sd in ("right", "left"):
            m = side == sd
            bvh = V[sd][1]
            tgt[m] = [bvh.find_nearest(Vector(q))[0] for q in Q[m]]
        s, R, t = umeyama(P, tgt)
    Q = (s * (R @ P.T)).T + t
    res = np.linalg.norm(Q - tgt, axis=1)
    return s, R, t, Q, tgt, res, V


def hex_cap(bm, centre, normal, radius):
    n = Vector(normal).normalized()
    a = n.orthogonal().normalized(); b = n.cross(a)
    ring = [bm.verts.new(centre + radius * (math.cos(k * math.pi / 3) * a + math.sin(k * math.pi / 3) * b))
            for k in range(6)]
    return bm.faces.new(ring)


def build():
    s, R, t, Q, tgt, res, V = fit()
    print(f"fit: scale {s:.3f} (micro-CT fly -> NeuroMechFly), "
          f"rotation {math.degrees(math.acos(min(1, (np.trace(R) - 1) / 2))):.1f} deg, "
          f"residual median {np.median(res)*1000:.1f} um, 95% {np.percentile(res,95)*1000:.1f} um")
    coll = bpy.data.collections["fly_body"]
    mats = {}
    for name, rgba in (("yellow", (0.95, 0.75, 0.15, 1)), ("pale", (0.55, 0.8, 1.0, 1)),
                       ("DRA", (0.8, 0.2, 0.9, 1))):
        m = bpy.data.materials.get(f"facet_{name}") or bpy.data.materials.new(f"facet_{name}")
        m.use_nodes = True
        bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.3
        m.diffuse_color = rgba
        mats[name] = m
    placement = []
    for sd in ("right", "left"):
        for nm in (f"ommatidia_{sd}", f"optical_axes_{sd}"):
            if nm in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
        bvh = V[sd][1]
        bm, bl = bmesh.new(), bmesh.new()
        fid = bm.faces.layers.int.new("facet_id")
        idx = [i for i, f in enumerate(FACETS) if f["side"] == sd]
        for i in idx:
            f = FACETS[i]
            loc, nrm, _, _ = bvh.find_nearest(Vector(Q[i]))
            d = Vector((R @ np.array(f["dir"])).tolist()).normalized()
            if nrm.dot(d) < 0:
                nrm = -nrm
            face = hex_cap(bm, loc + nrm * 0.002, nrm, 0.0085)
            face[fid] = f["id"]
            face.material_index = ("yellow", "pale", "DRA").index(f["subtype"])
            v0 = bl.verts.new(loc); v1 = bl.verts.new(loc + d * 0.25)
            bl.edges.new((v0, v1))
            placement.append({"id": f["id"], "side": sd,
                              "pos_head_mm": [round(c, 5) for c in loc],
                              "dir_head": [round(c, 5) for c in d],
                              "surface_normal_head": [round(c, 5) for c in nrm],
                              "axis_vs_normal_deg": round(math.degrees(d.angle(nrm)), 2),
                              "fit_residual_um": round(float(res[i]) * 1000, 2)})
        for nm, b in ((f"ommatidia_{sd}", bm), (f"optical_axes_{sd}", bl)):
            me = bpy.data.meshes.new(nm); b.to_mesh(me); b.free()
            ob = bpy.data.objects.new(nm, me); coll.objects.link(ob)
            ob.parent = head; ob.matrix_parent_inverse.identity(); ob.matrix_basis.identity()
            if nm.startswith("ommatidia"):
                for k in ("yellow", "pale", "DRA"):
                    me.materials.append(mats[k])
                ob["basis"] = "axes measured (micro-CT); positions approximate (fit to NMF eye)"
            else:
                ob.hide_render = True
                ob.color = (0.1, 0.9, 0.3, 1)
    (ROOT / "data/ommatidia_placement.json").write_text(json.dumps({
        "frame": "c_head body frame, mm (+x anterior, +y left, +z dorsal)",
        "fit": {"scale": s, "R": R.tolist(), "t_mm": t.tolist(),
                "residual_um_median": float(np.median(res) * 1000),
                "residual_um_p95": float(np.percentile(res, 95) * 1000)},
        "basis": {"position": "approximate (micro-CT lens positions fitted onto NeuroMechFly eye mesh)",
                  "direction": "measured (micro-CT, Zhao et al. 2025), rotated by the rigid part of the fit"},
        "facets": placement}))
    ang = [p["axis_vs_normal_deg"] for p in placement]
    print("optical axis vs mesh normal: median", round(float(np.median(ang)), 1), "deg")


build()
