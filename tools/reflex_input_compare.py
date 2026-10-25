"""Inputs to one silent-in-live premotor interneuron, offline vs live, in the resistance-reflex protocol.
usage:
  .venv/Scripts/python tools/reflex_input_compare.py probe <model_index>     -> recordings/probe_inputs_<id>.json
  (run the live protocol with that probe, output recordings/reflex_calib_inputs_<id>.csv)
  .venv-flygym/Scripts/python tools/reflex_input_compare.py offline <model_index>  -> offline partner rates
  .venv/Scripts/python tools/reflex_input_compare.py compare <model_index>  -> drive per partner, offline vs live"""
import sys, json, csv, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
mode, tid = sys.argv[1], int(sys.argv[2])

if mode == "probe":
    import pandas as pd
    e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    x = e[e.Postsynaptic_Index == tid].groupby("Presynaptic_Index")["Excitatory x Connectivity"].sum()
    g = {f"p{int(i)}": [int(i)] for i in x.index}; g[f"target_{tid}"] = [tid]
    json.dump(g, open(ROOT / f"recordings/probe_inputs_{tid}.json", "w"))
    x.to_csv(ROOT / f"recordings/inputs_{tid}_weights.csv")
    print(len(g), "groups")
elif mode == "offline":
    sys.path.insert(0, str(ROOT / "tools"))
    from vnc_body_model import Cord, Legs, LEGS
    li = LEGS.index("lf"); cord = Cord(); body = Legs(cord)
    pres = [int(r[0]) for r in list(csv.reader(open(ROOT / f"recordings/inputs_{tid}_weights.csv")))[1:]]
    AMP, F, T_SETTLE = 0.1745, 2.0, 0.5
    ang = np.zeros((6, 6)); om = np.zeros((6, 6)); lx = np.full(6, 0.5); acc = np.zeros(cord.n); n = 0
    for s in range(int((T_SETTLE + 1.0) / cord.dt)):
        t = s * cord.dt
        if t >= T_SETTLE:
            ph = 2 * np.pi * F * (t - T_SETTLE); ang[li, 4] = AMP * np.sin(ph); om[li, 4] = AMP * 2 * np.pi * F * np.cos(ph)
        cord.step(drive_on=False, clamp_idx=body.s_idx, clamp_val=body.afferents(ang, om, lx))
        if t >= T_SETTLE:
            acc += cord.R; n += 1
    R = acc / n
    with open(ROOT / f"recordings/inputs_{tid}_offline_rates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pre", "offline_hz"])
        for p in pres + [tid]:
            w.writerow([p, float(R[cord.pos[p]]) if p in cord.pos else float("nan")])
    print("offline target rate", float(R[cord.pos[tid]]))
else:
    import pandas as pd
    wts = pd.read_csv(ROOT / f"recordings/inputs_{tid}_weights.csv", index_col=0).iloc[:, 0]
    off = pd.read_csv(ROOT / f"recordings/inputs_{tid}_offline_rates.csv", index_col=0).offline_hz
    d = pd.read_csv(ROOT / f"recordings/reflex_calib_inputs_{tid}.csv"); on = d[d.t_ms > 4000]
    v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv").set_index("model_index")
    rows = []
    for p, w in wts.items():
        live = on[f"p{p}"].mean() if f"p{p}" in on else np.nan
        rows.append((p, str(v.cell_type.get(p, "brain")), str(v.super_class.get(p, "brain")), w, off.get(p, np.nan), live))
    r = pd.DataFrame(rows, columns=["pre", "type", "class", "syn", "offline_hz", "live_hz"])
    r["off_drive"] = r.syn * r.offline_hz; r["live_drive"] = r.syn * r.live_hz; r["diff"] = r.live_drive - r.off_drive
    print(f"target {tid}: offline {off.get(tid):.1f} Hz | live {on[f'target_{tid}'].mean():.1f} Hz")
    print(f"total drive (syn x Hz): offline excit {r.off_drive.clip(lower=0).sum():.0f} inhib {r.off_drive.clip(upper=0).sum():.0f} | "
          f"live excit {r.live_drive.clip(lower=0).sum():.0f} inhib {r.live_drive.clip(upper=0).sum():.0f}")
    print(r.groupby("class")[["off_drive", "live_drive"]].sum().round(0).to_string())
    print(r.sort_values("diff").head(8).round(1).to_string(index=False))
    print(r.sort_values("diff").tail(8).round(1).to_string(index=False))
