"""T4/T5 preferred directions for several FlyVis models (central column, moving edges, 12 directions).
Checks the anatomical co-tuning constraint: T4x and T5x share a lobula-plate layer and prefer the same
direction (Maisak et al. 2013). usage: pd_check.py 005 022 036 ..."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _env  # noqa: F401
import numpy as np, torch
from flyvis import NetworkView
from export_flyvis import edge_movie, DT

for m in sys.argv[1:]:
    net = NetworkView(f"flow/0000/{m}").init_network(); net.eval(); c = net.connectome
    types = [t.decode() for t in c.unique_cell_types[:]]
    nt = np.array([types.index(t.decode()) for t in c.nodes.type[:]]); u = np.asarray(c.nodes.u[:]); v = np.asarray(c.nodes.v[:])
    inp = net.stimulus.input_index; d = {"hex_u": u[inp[0]], "hex_v": v[inp[0]]}
    with torch.no_grad():
        st = net.steady_state(1.0, DT, 1)
    vg = st.nodes.activity.numpy().reshape(-1)
    th = np.radians(np.arange(0, 360, 30)); pd = {}
    for on in (True, False):
        R = []
        with torch.no_grad():
            for a in th:
                R.append(net.simulate(torch.tensor(edge_movie(d, a, on))[None, :, None], DT, initial_state=st).numpy()[0])
        for s in "abcd":
            t = ("T4" if on else "T5") + s
            k = int(np.where((nt == types.index(t)) & (u == 0) & (v == 0))[0][0])
            r = np.array([np.maximum(x[:, k], 0).max() - max(vg[k], 0) for x in R]).clip(0)
            pd[t] = np.degrees(np.angle((r * np.exp(1j * th)).sum())) % 360
    diff = {s: round(float(abs((pd["T4" + s] - pd["T5" + s] + 180) % 360 - 180))) for s in "abcd"}
    print(f"{m}: " + " ".join(f"{k} {v:.0f}" for k, v in pd.items()) + f" | T4-T5 mismatch {diff} max {max(diff.values())}", flush=True)
