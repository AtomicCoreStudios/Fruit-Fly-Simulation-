"""Screen the 50 pretrained FlyVis models on established T4/T5 physiology, independent of any loom
result: T4/T5 are strongly direction selective (Maisak et al. 2013 Nature 500:212) and respond much
more to moving edges than to stationary flashes. For each model:
  DSI     mean direction-selectivity index of T4a-d (ON edges) and T5a-d (OFF edges), central column
  flash   mean ratio (response to a stationary 5-column flash of the cell's polarity) /
          (response to its preferred-direction moving edge); lower = more motion-specific
  T2off   T2 response to an OFF flash (informative only)
Writes recordings/flyvis_ensemble_screen.csv and prints the ranking (score = DSI - flash).
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _env  # noqa: F401
import numpy as np, pandas as pd, torch
from flyvis import NetworkView
from export_flyvis import edge_movie, DT

ROOT = _env.ROOT
TYPES = [f"T{k}{s}" for k in "45" for s in "abcd"]


def screen(model):
    net = NetworkView(model).init_network(); net.eval()
    c = net.connectome
    types = [t.decode() for t in c.unique_cell_types[:]]
    ntype = np.array([types.index(t.decode()) for t in c.nodes.type[:]])
    u = np.asarray(c.nodes.u[:]); v = np.asarray(c.nodes.v[:])
    inp = net.stimulus.input_index
    d = {"hex_u": u[inp[0]], "hex_v": v[inp[0]]}
    with torch.no_grad():
        st = net.steady_state(1.0, DT, 1)
    vg = st.nodes.activity.numpy().reshape(-1)
    run = lambda m: net.simulate(torch.tensor(m, dtype=torch.float32)[None, :, None], DT, initial_state=st).numpy()[0]
    cen = {t: int(np.where((ntype == types.index(t)) & (u == 0) & (v == 0))[0][0]) for t in TYPES + ["T2"]}
    x = 1.5 * d["hex_v"]; y = -np.sqrt(3) * (d["hex_u"] + d["hex_v"] / 2); r = np.hypot(x, y)
    th = np.radians(np.arange(0, 360, 45))
    resp = {t: [] for t in TYPES}
    with torch.no_grad():
        for on in (True, False):
            for a in th:
                R = run(edge_movie(d, a, on))
                for t in TYPES:
                    if t.startswith("T4") == on:
                        resp[t].append(np.maximum(R[:, cen[t]], 0).max() - max(vg[cen[t]], 0))
        fl = {}
        for on in (True, False):
            m = np.full((200, len(r)), 0.5, np.float32); m[100:, r <= 5 * np.sqrt(3)] = 1.0 if on else 0.0
            fl[on] = run(m)
    dsi, flash = [], []
    for t in TYPES:
        rr = np.maximum(np.array(resp[t]), 0)
        vec = (rr * np.exp(1j * th)).sum(); dsi.append(abs(vec) / max(rr.sum(), 1e-9))
        f = np.maximum(fl[t.startswith("T4")][100:, cen[t]], 0).max() - max(vg[cen[t]], 0)
        flash.append(max(f, 0) / max(rr.max(), 1e-9))
    t2off = np.maximum(fl[False][100:, cen["T2"]], 0).max() - max(vg[cen["T2"]], 0)
    return float(np.mean(dsi)), float(np.mean(flash)), float(t2off)


def main():
    rows = []
    for k in range(50):
        m = f"flow/0000/{k:03d}"
        try:
            dsi, fl, t2 = screen(m)
        except Exception as e:  # noqa: BLE001
            print(m, "failed:", e); continue
        rows.append((m, dsi, fl, t2, dsi - fl))
        print(f"{m}: DSI {dsi:.2f}  flash/edge {fl:.2f}  T2 OFF {t2:+.2f}", flush=True)
    df = pd.DataFrame(rows, columns=["model", "dsi", "flash_ratio", "t2_off", "score"]).sort_values("score", ascending=False)
    df.to_csv(ROOT / "recordings/flyvis_ensemble_screen.csv", index=False)
    print("\nbest by score (DSI - flash ratio):\n", df.head(8).round(3).to_string(index=False))
    print("\nmodel 000 rank:", int(np.where(df.model.values == "flow/0000/000")[0][0]) + 1, "of", len(df))


if __name__ == "__main__":
    main()
