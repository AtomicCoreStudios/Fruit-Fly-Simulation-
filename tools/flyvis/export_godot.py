"""Pack the FlyVis network + eye mapping for Godot: data/flyvis_gpu.bin and data/flyvis_gpu.json.

Godot (shaders/flyvis.glsl) runs two instances (right and left eye) of the same network. Each
FlyVis column reads the luminance of its facet, and every FlyVis cell whose type also exists in the
FlyWire column map drives the real FlyWire neuron(s) of that type in the matching column with a
Poisson rate:  rate = gain * max(relu(V) - relu(V_grey), 0)
(V_grey: activity at the grey-adapted steady state). basis: approximate (rate read-out is a free
parameter; FlyVis activity is unitless).
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    net = np.load(ROOT / "data/flyvis_net.npz")
    mp = json.loads((ROOT / "data/flyvis_map.json").read_text())["eyes"]
    om = json.loads((ROOT / "data/ommatidia.json").read_text())["facets"]
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    index = pd.Series(np.arange(len(comp)), index=comp.index.astype(np.int64))
    types = list(net["types"]); N = len(net["u"])
    cols = pd.read_csv(ROOT / "data/raw/column_assignment.csv.gz")[["root_id", "hemisphere", "type", "p", "q"]]
    extra = ROOT / "data/extra_columns.csv"   # tools/flyvis/assign_columns.py (connectivity-based, approximate)
    if extra.exists():
        ex = pd.read_csv(extra)
        cols = pd.concat([cols, ex[ex.type != "Am"][["root_id", "hemisphere", "type", "p", "q"]]])
    shared = sorted(set(cols.type) & set(types) - {"R7", "R8"})
    by_col = cols[cols.type.isin(shared)].groupby(["hemisphere", "p", "q", "type"]).root_id.apply(list).to_dict()
    node_at = {(int(t), int(u), int(v)): i for i, (t, u, v) in enumerate(zip(net["ntype"], net["u"], net["v"]))}
    in_facet = np.full(2 * N, -1, np.int32)
    pairs, base = [], []
    vgrey_relu = np.maximum(net["v_grey"], 0)
    stats = {}
    for e, side in enumerate(("right", "left")):
        m = mp[side]
        for k, (hu, hv) in enumerate(zip(net["hex_u"], net["hex_v"])):
            f = m["hexal_facet"][k]
            for t in range(net["input_index"].shape[0]):
                in_facet[e * N + net["input_index"][t][k]] = f
            p, q = m["hexal_pq"][k]
            for t in shared:
                node = node_at.get((types.index(t), int(hu), int(hv)))
                if node is None:   # strided FlyVis type (e.g. Lawf, Mi15): nearest cell of that type
                    k_t = types.index(t)
                    cand = np.where(net["ntype"] == k_t)[0]
                    if len(cand) == 0:
                        continue
                    du_ = net["u"][cand] - hu; dv_ = net["v"][cand] - hv
                    node = int(cand[np.argmin(np.abs(du_) + np.abs(dv_) + np.abs(du_ + dv_))])
                for rid in by_col.get((side, p, q, t), []):
                    if int(rid) in index.index:
                        pairs.append((int(index[int(rid)]), e * N + node)); base.append(vgrey_relu[node])
                        stats[t] = stats.get(t, 0) + 1
    pairs = np.array(pairs, np.int32); base = np.array(base, np.float32)
    # FlyVis is exactly convolutional: every cell of a type shares one filter of
    # (source type, du, dv, weight). 2,355 entries reproduce all 1.5M edges (checked below).
    nt, u, v = net["ntype"], net["u"], net["v"]
    tar = np.repeat(np.arange(N), np.diff(net["row"])); src = net["src"]
    key = np.stack([nt[tar], nt[src], u[tar] - u[src], v[tar] - v[src]], 1)
    uk, first = np.unique(key, axis=0, return_index=True)
    fw = net["w"][first].astype(np.float32)
    order = np.lexsort((uk[:, 1], uk[:, 0])); uk, fw = uk[order], fw[order]
    filt_ptr = np.searchsorted(uk[:, 0], np.arange(len(types) + 1)).astype(np.int32)
    filt = np.zeros((len(uk), 4), np.int32)
    filt[:, :3] = uk[:, 1:]; filt[:, 3] = fw.view(np.int32)
    G = 2 * 15 + 1
    grid = np.full(len(types) * G * G, -1, np.int32)
    grid[nt * G * G + (u + 15) * G + (v + 15)] = np.arange(N)
    n_rec = 0
    for T in range(len(types)):
        cells = np.where(nt == T)[0]
        for S, du, dv in uk[filt_ptr[T]:filt_ptr[T + 1], 1:]:
            su, sv = u[cells] - du, v[cells] - dv
            ok = (np.abs(su) <= 15) & (np.abs(sv) <= 15) & (np.abs(su + sv) <= 15)
            n_rec += int((grid[S * G * G + (su[ok] + 15) * G + (sv[ok] + 15)] >= 0).sum())
    assert n_rec == len(net["w"]), (n_rec, len(net["w"]))
    cell = np.stack([nt, u, v, np.zeros_like(nt)], 1).astype(np.int32)
    blobs = [("bias", net["bias"].astype(np.float32)), ("tau", net["tau"].astype(np.float32)),
             ("v_grey", net["v_grey"].astype(np.float32)), ("cell", cell.reshape(-1)),
             ("filt_ptr", filt_ptr), ("filt", filt.reshape(-1)), ("grid", grid),
             ("in_facet", in_facet), ("out_pairs", pairs.reshape(-1)), ("out_base", base)]
    meta = {"model": str(net["model"]), "n_nodes": N, "n_edges": int(len(net["w"])),
            "n_out": int(len(pairs)), "n_filter": int(len(uk)), "extent": 15, "dt_s": 1 / 200, "types": types, "shared_types": shared,
            "node_type": net["ntype"].tolist(), "offsets": {}, "driven_per_type": stats,
            "basis": {"connectivity": "measured (FIB-25/FIB-19 medulla, per-type average filters)",
                      "parameters": "model-fitted (FlyVis optic-flow task, Lappalainen et al. 2024)",
                      "lattice_on_eye": "approximate (oriented by T4/T5 preferred directions)",
                      "rate_readout": "approximate (free gain)"}}
    off = 0
    with open(ROOT / "data/flyvis_gpu.bin", "wb") as fh:
        for name, arr in blobs:
            b = arr.tobytes()
            meta["offsets"][name] = [off, len(b)]
            fh.write(b); off += len(b)
    (ROOT / "data/flyvis_gpu.json").write_text(json.dumps(meta))
    print(f"wrote data/flyvis_gpu.bin ({off/1e6:.1f} MB): {N} cells x 2 eyes, {len(net['w'])} edges, "
          f"{len(pairs)} FlyWire neurons driven across {len(stats)} types")
    print(" ", stats)


if __name__ == "__main__":
    main()
