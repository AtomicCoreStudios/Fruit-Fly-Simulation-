"""Pack the graded network for Godot (shaders/graded.glsl):
  data/graded_gpu.bin / .json  graded units, CSR inputs, FlyVis gather map
  data/lif_w_graded.bin        copy of the LIF weight array (int32, CSR order of
                               data/connectome_flywire.bin) with the synapses that the graded model
                               now carries set to 0: pre in (FlyVis-driven | graded non-VPN |
                               photoreceptors), post in (FlyVis-driven | graded). VPN outputs (axonal,
                               spiking) and all central-brain synapses are untouched.
"""
import json, pathlib, struct
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    g = np.load(ROOT / "data/graded_net.npz", allow_pickle=True)
    nD, nG = len(g["D_idx"]), len(g["G_idx"])
    blobs = [("tau", g["tau"].astype(np.float32)), ("bias", g["bias"].astype(np.float32)),
             ("row_ptr", g["row_ptr"].astype(np.int32)), ("col", g["col"].astype(np.int32)),
             ("w", g["w"].astype(np.float32)), ("d_node", g["D_node"].astype(np.int32)),
             ("g_fw", g["G_idx"].astype(np.int32)), ("g_vpn", g["G_vpn"].astype(np.int32))]
    meta = {"n_d": nD, "n_g": nG, "n_edges": int(len(g["w"])), "dt_s": 1 / 200, "offsets": {},
            "alpha": float(g["alpha"]), "beta": float(g["beta"]), "kappa": float(g["kappa"]),
            "vpn_types": sorted(set(g["G_type"][g["G_vpn"]].tolist())),
            "basis": "connectivity measured (FlyWire v783); strengths FlyVis-derived (per presynaptic "
                     "type median, in-degree rule, kappa, global stability scale alpha); see "
                     "tools/graded/build_graded.py"}
    off = 0
    with open(ROOT / "data/graded_gpu.bin", "wb") as fh:
        for name, arr in blobs:
            b = arr.tobytes(); meta["offsets"][name] = [off, len(b)]; fh.write(b); off += len(b)
    (ROOT / "data/graded_gpu.json").write_text(json.dumps(meta))
    print(f"graded_gpu.bin {off/1e6:.1f} MB: {nG} graded units ({int(g['G_vpn'].sum())} VPN dendrites), "
          f"{len(g['w'])} edges, {nD} FlyVis-driven inputs")

    # ---- LIF weights with graded-carried synapses silenced
    f = open(ROOT / "data/connectome_flywire.bin", "rb")
    assert f.read(4) == b"FLYC"
    _, n, e, n_pad, n_groups = struct.unpack("<iiiii", f.read(20))
    f.seek(24 + 2 * n * 4)
    row_ptr = np.frombuffer(f.read((n + 1) * 4), np.int32)
    col = np.frombuffer(f.read(e * 4), np.int32)
    w_off = f.tell()
    w = np.frombuffer(f.read(e * 4), np.int32).copy()
    pre = np.repeat(np.arange(n), np.diff(row_ptr))
    vt = pd.read_csv(ROOT / "data/raw/visual_neuron_types.csv.gz")
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    index = pd.Series(np.arange(n), index=comp.index.astype(np.int64))
    pr = np.zeros(n, bool)
    pr[index.reindex(vt[vt.type.isin(["R1-6", "R7", "R8"])].root_id).dropna().astype(int).values] = True
    sil_pre = g["silence_pre"] | pr
    m = sil_pre[pre] & g["silence_post"][col]
    w[m] = 0
    w.tofile(ROOT / "data/lif_w_graded.bin")
    meta2 = {"w_offset_in_connectome": int(w_off), "n_edges": int(e), "silenced": int(m.sum())}
    (ROOT / "data/lif_w_graded.json").write_text(json.dumps(meta2))
    print(f"lif_w_graded.bin: {m.sum()} of {e} LIF edges silenced (carried by the graded model)")


if __name__ == "__main__":
    main()
