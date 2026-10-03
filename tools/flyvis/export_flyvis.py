"""Export a pretrained FlyVis network (Lappalainen et al. 2024, Nature; github.com/TuragaLab/flyvis,
MIT) so Godot can run it on the GPU, and verify the export.

FlyVis: 65 optic-lobe cell types on a hexagonal lattice of 721 columns (extent 15), connectivity
from the FIB-25/FIB-19 medulla reconstructions averaged into per-type convolutional filters, signs
from transmitters; per-type resting potential and time constant and per-type-pair synapse strength
were FITTED so the network computes optic flow. Dynamics (PPNeuronIGRSynapses):
    tau_i dV_i/dt = -V_i + bias_i + sum_j w_ij * relu(V_j) + x_i      (x_i: stimulus on R1-R8)
basis: connectivity measured (FIB-25, a different fly from FlyWire); parameters model-fitted.

Outputs data/flyvis_net.npz and prints:
  1. max |difference| between this numpy re-implementation and FlyVis's own simulate()
  2. preferred direction of each T4/T5 subtype (central column) to moving ON/OFF edges, in the
     FlyVis pixel frame (x = 1.5 v, y = -sqrt(3) (u + v/2)), used to orient the lattice on the eye.

usage: .venv/Scripts/python tools/flyvis/export_flyvis.py [--model flow/0000/000]
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _env  # noqa: F401  (sets FLYVIS_ROOT_DIR, patches datamate on Windows)
import numpy as np, torch
from flyvis import NetworkView

ROOT = _env.ROOT
DT = 1 / 200  # s; FlyVis analyses simulate at 5 ms (training used 20 ms)


def export(model):
    nv = NetworkView(model)
    net = nv.init_network()
    net.eval()
    c = net.connectome
    p = net._param_api()
    types = [t.decode() for t in c.unique_cell_types[:]]
    ntype = np.array([types.index(t.decode()) for t in c.nodes.type[:]], np.int32)
    u = np.asarray(c.nodes.u[:], np.int32); v = np.asarray(c.nodes.v[:], np.int32)
    bias = p.nodes.bias.detach().numpy().astype(np.float32)
    tau = p.nodes.time_const.detach().numpy().astype(np.float32)
    src = np.asarray(c.edges.source_index[:], np.int64)
    tar = np.asarray(c.edges.target_index[:], np.int64)
    w = p.edges.weight.detach().numpy().astype(np.float32)
    order = np.argsort(tar, kind="stable")
    src, tar, w = src[order], tar[order], w[order]
    row = np.zeros(len(u) + 1, np.int64)
    np.cumsum(np.bincount(tar, minlength=len(u)), out=row[1:])
    inp = net.stimulus.input_index                          # (8 types, n_hexals) node indices
    hex_u, hex_v = u[inp[0]], v[inp[0]]
    # steady state under uniform grey (0.5), as FlyVis does before every stimulus
    with torch.no_grad():
        st = net.steady_state(1.0, DT, 1)
    v_grey0 = st.nodes.activity.detach().numpy().reshape(-1).astype(np.float32)
    # FlyVis' 1 s steady state still drifts ~0.01; settle 3 s more under grey for the resting state
    with torch.no_grad():
        grey = torch.full((1, int(3.0 / DT), 1, inp_n := net.stimulus.input_index.shape[1]), 0.5)
        v_grey = net.simulate(grey, DT, initial_state=st).numpy()[0, -1].astype(np.float32)
    return net, dict(types=np.array(types), ntype=ntype, u=u, v=v, bias=bias, tau=tau,
                     row=row.astype(np.int32), src=src.astype(np.int32), w=w,
                     input_index=inp.astype(np.int32), hex_u=hex_u, hex_v=hex_v, v_grey=v_grey, v_grey0=v_grey0)


def numpy_sim(d, movie, v0):
    """Euler integration identical to PPNeuronIGRSynapses; movie: (T, n_hexals)."""
    V = v0.copy()
    tau = np.maximum(d["tau"], DT)
    rows = np.repeat(np.arange(len(V)), np.diff(d["row"]))
    out = []
    for x in movie:
        xin = np.zeros_like(V)
        for k in range(d["input_index"].shape[0]):
            xin[d["input_index"][k]] += x
        syn = np.bincount(rows, weights=d["w"] * np.maximum(V[d["src"]], 0), minlength=len(V))
        V = V + DT / tau * (-V + d["bias"] + syn + xin)
        out.append(V.copy())
    return np.array(out)


def edge_movie(d, theta, on, speed_cols=10.0):
    """Edge moving across a grey (0.5) background in the FlyVis pixel frame at `speed_cols`
    columns/s (~48 deg/s at 4.8 deg per column); ON edge -> 1.0 behind it, OFF edge -> 0.0."""
    x = 1.5 * d["hex_v"]; y = -np.sqrt(3) * (d["hex_u"] + d["hex_v"] / 2)
    proj = x * np.cos(theta) + y * np.sin(theta)
    v_px = speed_cols * np.sqrt(3)
    n = int(((proj.max() - proj.min()) + 4) / v_px / DT) + 40
    t = np.arange(n) * DT
    front = proj.min() - 2 + v_px * t
    passed = proj[None, :] < front[:, None]
    return np.where(passed, 1.0 if on else 0.0, 0.5).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="flow/0000/000")
    a = ap.parse_args()
    net, d = export(a.model)
    print(f"{a.model}: {len(d['u'])} cells, {len(d['w'])} edges, {len(d['types'])} types, "
          f"{d['input_index'].shape[1]} columns")
    # 1. verify against FlyVis
    rng = np.random.default_rng(0)
    movie = np.clip(0.5 + 0.3 * rng.standard_normal((40, d["input_index"].shape[1])), 0, 1).astype(np.float32)
    with torch.no_grad():
        st = net.steady_state(1.0, DT, 1)
        ref = net.simulate(torch.tensor(movie)[None, :, None], DT, initial_state=st).numpy()[0]
    mine = numpy_sim(d, movie, d["v_grey0"])
    print(f"verification vs FlyVis simulate(): max |dV| = {np.abs(ref - mine).max():.2e} "
          f"(activity range {ref.min():.2f}..{ref.max():.2f})")
    # 2. T4/T5 preferred directions (central column)
    thetas = np.radians(np.arange(0, 360, 30))
    centre = {t: int(np.where((d["ntype"] == list(d["types"]).index(t)) & (d["u"] == 0) & (d["v"] == 0))[0][0])
              for t in [f"T{k}{s}" for k in "45" for s in "abcd"]}
    pd = {}
    for t, idx in centre.items():
        on = t.startswith("T4")
        resp = []
        for th in thetas:
            m = edge_movie(d, th, on)
            with torch.no_grad():
                r = net.simulate(torch.tensor(m)[None, :, None], DT, initial_state=st).numpy()[0][:, idx]
            resp.append(np.maximum(r, 0).max() - max(d["v_grey"][idx], 0))
        resp = np.array(resp)
        vec = (resp * np.exp(1j * thetas)).sum()
        dsi = abs(vec) / max(resp.sum(), 1e-9)
        pd[t] = (float(np.degrees(np.angle(vec)) % 360), float(dsi))
        print(f"  {t}: preferred direction {pd[t][0]:6.1f} deg (FlyVis pixel frame), DSI {dsi:.2f}, "
              f"peak {resp.max():.3f}")
    np.savez_compressed(ROOT / "data/flyvis_net.npz", **d,
                        pd_types=np.array(list(pd)), pd_deg=np.array([v[0] for v in pd.values()]),
                        pd_dsi=np.array([v[1] for v in pd.values()]), model=np.array(a.model))
    print("wrote data/flyvis_net.npz")


if __name__ == "__main__":
    main()
