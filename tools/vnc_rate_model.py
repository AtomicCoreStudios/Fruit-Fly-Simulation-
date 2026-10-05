"""Reproduction of the Pugliese et al. 2025 VNC rate model (bioRxiv 10.1101/2025.09.12.675944;
github.com/smpuglie/Pugliese_2026, src/simulation/vnc_sim.py, configs/neuron_params/default.yaml) on OUR nerve-cord
graph (BANC VNC synapses, data/vnc/vnc_edges.parquet, incl. bridged DNs/ANs):
  dR/dt = ( max(0, fcap * tanh((a/fcap) * (I + W R - theta))) - R ) / tau
  tau 20 ms, a 1, theta 7.5, fcap 200 Hz (paper means; they also sample +-10% per replicate), W = 0.03 x signed
  synapse count; a /= size, theta *= size with size = total synapse count / median (stand-in, see below); input 250 to
  the stimulated DN. Euler, dt 0.25 ms.
usage: vnc_rate_model.py [DN model index ...]   (default: left DNg100)"""
import sys, json, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = pathlib.Path(__file__).resolve().parents[1]
e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet")
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
b = pd.read_csv(ROOT / "data/vnc/vnc_bridge.csv")
import os
SUB = os.environ.get("RM_SUBNET", "front_left")   # paper: front-leg MNs + their presynaptic partners + the DN
SIZE = os.environ.get("RM_SIZE", "sqrt")          # none | sqrt | linear (size proxy = total synapse count)
fl = v[v.cell_class.astype(str).str.contains("leg_motor") & (v.sub_class == "front_leg_motor_neuron") & (v.side == "left")].model_index.values
if SUB == "front_left":
    keep = set(fl) | set(e[e.Postsynaptic_Index.isin(fl)].Presynaptic_Index) | {130297, 135733}
    e = e[e.Presynaptic_Index.isin(keep) & e.Postsynaptic_Index.isin(keep)]
ids = np.unique(np.concatenate([e.Presynaptic_Index.values, e.Postsynaptic_Index.values]))
pos = {x: i for i, x in enumerate(ids)}; n = len(ids)
pre = np.array([pos[x] for x in e.Presynaptic_Index]); post = np.array([pos[x] for x in e.Postsynaptic_Index])
W = sp.csr_matrix((0.03 * e["Excitatory x Connectivity"].astype(float).values, (post, pre)), shape=(n, n))
# sizes: BANC morphology columns are empty in our export, so size = total VNC synapses (in + out) / median
# (synapse number scales with neuron size; approximate stand-in for the paper's size measure)
cnt = np.bincount(pre, weights=e.Connectivity.values, minlength=n) + np.bincount(post, weights=e.Connectivity.values, minlength=n)
size = cnt.astype(float); size[size <= 0] = np.median(size[size > 0]); size /= np.median(size)
size = np.ones(n) if SIZE == "none" else (np.sqrt(size) if SIZE == "sqrt" else size)
if SIZE == "measured":
    # BANC surface areas from Pugliese et al.'s table (data/raw/pugliese) where available, median elsewhere
    pt = pd.read_csv(ROOT / "data/raw/pugliese/wTable_20260217_fullData_consistentColumns.csv")
    sa = dict(zip(pt.pt_root_id.astype("int64"), pt.surf_area_um2))
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    fwp = pd.Series(np.arange(len(comp)), index=comp.index)
    bid_of = dict(zip(v.model_index, v.bid))
    bid_of.update({int(fwp[f]): bb for f, bb in zip(b.flywire_id, b.banc_id) if f in fwp.index})
    size = np.array([sa.get(bid_of.get(x, -1), np.nan) for x in ids], float)
    print("neurons with measured surface area:", int(np.isfinite(size).sum()), "of", n)
    med = np.nanmedian(size); size[~np.isfinite(size)] = med; size /= med
a = 1.0 / size; theta = 7.5 * size; fcap, tau = 200.0, 0.020
stim = [int(x) for x in sys.argv[1:]] or [130297]
I = np.zeros(n); I[[pos[s] for s in stim]] = float(os.environ.get("RM_STIM_I", "400"))
dt, T = 0.00025, 2.0; R = np.zeros(n)
# record: CPG cells and front-left leg motor pools
rec = {}
for name, t in [("E1", "IN17A001"), ("E2", "INXXX466"), ("I1", "IN16B036"), ("I2", "IN19A007")]:
    for r_ in v[(v.cell_type == t) & (v.side == "left")].itertuples():
        rec[f"{name}_{r_.model_index}"] = [pos[r_.model_index]] if r_.model_index in pos else []
lm = v[v.cell_class.astype(str).str.contains("leg_motor") & (v.sub_class == "front_leg_motor_neuron") & (v.side == "left")]
for act, g in lm.groupby(lm.function.astype(str).str.replace("leg_motor", "").str.strip(",")):
    rec["MN_" + act] = [pos[x] for x in g.model_index if x in pos]
tr = {k: [] for k in rec}; ts = []
TAU_S = float(os.environ.get("RM_TAU_SYN", "0"))      # ms; live LIF: 5 ms synaptic filter
DELAY = int(round(float(os.environ.get("RM_DELAY", "0")) / (dt * 1000)))   # ms; live: 1.8 ms
syn = np.zeros(n); hist = [np.zeros(n)] * (DELAY + 1)
for s in range(int(T / dt)):
    hist.append(R.copy()); Rd = hist.pop(0) if DELAY > 0 else R
    drive = W @ Rd
    if TAU_S > 0:
        syn += dt * 1000 * (drive - syn) / TAU_S
        drive = syn
    x = I + drive - theta
    act = np.maximum(fcap * np.tanh((a / fcap) * x), 0.0)
    R += dt * (act - R) / tau
    if s % 4 == 0:
        ts.append(s * dt)
        for k, ix in rec.items():
            tr[k].append(R[ix].mean() if ix else 0.0)
d = pd.DataFrame(tr); d.insert(0, "t_ms", np.array(ts) * 1000); d.insert(1, "threat", 0)
out = ROOT / f"recordings/vnc_rate_model_{SUB}_{SIZE}{os.environ.get('RM_TAG', '')}.csv"; d.to_csv(out, index=False)
print(f"{n} neurons, {W.nnz} edges; active (>1 Hz) at end: {(R > 1).sum()}; saved {out.name}")
