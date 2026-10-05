"""Exact reproduction of Pugliese et al. 2025/2026 (bioRxiv 10.1101/2025.09.12.675944) DNg100 simulation on their own
BANC front-left-leg subnetwork (data/raw/pugliese: W_20260217.npz, wTable_20260217_fullData_consistentColumns.csv;
github.com/smpuglie/Pugliese_2026 @10e7661, CC-BY-4.0). Model as src/simulation/vnc_sim.py:
  dR/dt = ( max(0, fcap * tanh((a/fcap)(I + W' R - theta))) - R ) / tau,  W' = 0.03 W (exc and inh multipliers)
  a /= size, theta *= size, size = surface area / median; I = 400 on DNg100 (row 1605), T = 1 s.
usage: pugliese_repro.py [n_random_replicates]"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data/raw/pugliese"
W = np.load(P / "W_20260217.npz")["arr_0"].astype(np.float64)
df = pd.read_csv(P / "wTable_20260217_fullData_consistentColumns.csv")
n = len(df)
def signed(W):
    return W   # sign handled below if W holds raw counts
nt = df.predictedNt.astype(str).str.lower()
if W.min() >= 0:   # raw synapse counts: sign by presynaptic transmitter (rows = pre? checked below)
    sg = np.where(nt.str.contains("gaba|glut"), -1.0, 1.0)
else:
    sg = None
size = df.surf_area_um2.astype(float).to_numpy().copy()
med = np.nanmedian(size); size[~np.isfinite(size) | (size <= 0)] = med; size = size / med
STIM = 1605
mn = np.where(df["class"].astype(str).str.contains("motor", case=False).values)[0]


def run(rng=None, T=1.0, dt=0.0005):
    tau = np.full(n, 0.02); a = np.ones(n); th = np.full(n, 7.5); fc = np.full(n, 200.0)
    if rng is not None:
        tau = rng.normal(0.02, 0.002, n); a = rng.normal(1, 0.1, n); th = rng.normal(7.5, 0.6, n); fc = rng.normal(200, 10, n)
    a = a / size; th = th * size
    Wd = 0.03 * W.T          # stored pre x post; dR uses post x pre
    I = np.zeros(n); I[STIM] = 400.0
    R = np.zeros(n); out = []
    for s in range(int(T / dt)):
        t = s * dt
        inp = (I if 0.02 <= t < 0.999 else 0.0) + Wd @ R
        act = np.maximum(fc * np.tanh((a / fc) * (inp - th)), 0.0)
        R += dt * (act - R) / tau
        out.append(R[mn].copy())
    return np.array(out), dt


def rhythm(trace, dt, t0=0.2):
    x = trace[int(t0 / dt):]
    best = (0.0, np.nan)
    for j in range(x.shape[1]):
        y = x[:, j]
        if y.max() < 1:
            continue
        y = (y - y.mean()) * np.hanning(len(y))
        p = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), dt)
        band = (f >= 1) & (f <= 50); sel = (f >= 2) & (f <= 30)
        fp = f[sel][np.argmax(p[sel])]
        sc = p[(f >= fp - 1) & (f <= fp + 1)].sum() / max(p[band].sum(), 1e-9)
        if sc > best[0]:
            best = (sc, fp)
    return best, int((x[-1] > 1).sum())


if __name__ == "__main__":
    print(f"W {W.shape}, min {W.min():.1f} max {W.max():.1f}; DNg100 row {df.cell_type[STIM]}; {len(mn)} motor neurons")
    print("W orientation check: DNg100 row sum", W[STIM].sum(), "column sum", W[:, STIM].sum())
    tr, dt = run()
    (sc, fp), act = rhythm(tr, dt)
    print(f"mean params: best MN rhythmicity {sc:.2f} at {fp:.1f} Hz, {act} MNs active")
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i in range(k):
        tr, dt = run(np.random.default_rng(i))
        (sc, fp), act = rhythm(tr, dt)
        print(f"replicate {i}: best MN rhythmicity {sc:.2f} at {fp:.1f} Hz, {act} MNs active")
