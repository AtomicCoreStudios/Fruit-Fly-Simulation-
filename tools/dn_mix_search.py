"""Search the descending-neuron input mix that gives the most fly-like stepping in the offline VNC rate model
(tools/vnc_rate_model.py setup: Pugliese et al. 2025 equations, our BANC graph, real sizes, 5 ms synapse, 1.8 ms
delay). Inputs are restricted to DN types with documented walking roles and are left/right symmetric.
Score per leg (all six): swing pool (coxa promotion + trochanter flexion) and stance pool (coxa remotion +
trochanter extension) both modulated, their phase difference near 180 deg, frequency in 7-15 Hz (stepping of
walking flies); plus tripod coordination (L1,R2,L3 vs R1,L2,R3 in anti-phase). Random search, fixed seed.
usage: dn_mix_search.py [n_trials]   -> recordings/dn_mix_search.csv"""
import os, sys, json, pathlib, itertools
import numpy as np, pandas as pd
os.environ.setdefault("RM_SUBNET", "full"); os.environ.setdefault("RM_SIZE", "measured")
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "tools/vnc_rate_model.py").read_text(encoding="utf-8")
setup = src[:src.index("# stimulus:")]
g = {"__file__": str(ROOT / "tools/vnc_rate_model.py")}; sys_argv = sys.argv; sys.argv = ["x"]
exec(compile(setup, "vnc_rate_model_setup", "exec"), g); sys.argv = sys_argv
W, pos, n, a, theta, fcap, tau = g["W"].tocsr(), g["pos"], g["n"], g["a"], g["theta"], g["fcap"], g["tau"]
lm = json.load(open(ROOT / "data/leg_motor_map.json"))["legs"]
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
pools = {}
for leg in LEGS:
    m = lm[leg]["motor"]
    sw = [x for x in m.get("ThC_pro", {}).get("pos", []) + m.get("CTr", {}).get("pos", []) if x in pos]
    st = [x for x in m.get("ThC_pro", {}).get("neg", []) + m.get("CTr", {}).get("neg", []) if x in pos]
    pools[leg] = ([pos[x] for x in sw], [pos[x] for x in st])
DN = {"DNg100": [130297, 135733], "DNb08": [47118, 98130, 115273, 131703], "DNa02": [904, 92992],
      "DNg97": [42812, 78013], "DNp09": [83620, 119032]}
dt, T, DELAY, TAU_S = 0.00025, 1.5, int(round(1.8 / 0.25)), 5.0


def sim(amps):
    I = np.zeros(n)
    for k, amp in amps.items():
        for x in DN[k]:
            if x in pos:
                I[pos[x]] = amp
    R = np.zeros(n); syn = np.zeros(n); hist = [np.zeros(n)] * (DELAY + 1); rec = []
    for s in range(int(T / dt)):
        hist.append(R.copy()); Rd = hist.pop(0)
        syn += dt * 1000 * (W @ Rd - syn) / TAU_S
        R += dt * (np.maximum(fcap * np.tanh((a / fcap) * (I + syn - theta)), 0.0) - R) / tau
        if s % 4 == 0 and s * dt > 0.3:
            rec.append([R[ix].sum() if ix else 0.0 for leg in LEGS for ix in pools[leg]])
    return np.array(rec), dt * 4


def score(rec, dts):
    t = np.arange(len(rec)) * dts; res = {}; phases = {}; total = 0.0
    for k, leg in enumerate(LEGS):
        sw, st = rec[:, 2 * k], rec[:, 2 * k + 1]
        if sw.mean() < 0.5 or st.mean() < 0.5:
            res[leg] = (np.nan, 0, 0, np.nan); continue
        x = sw - sw.mean(); p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; f = np.fft.rfftfreq(len(x), dts)
        sel = (f >= 3) & (f <= 25); f0 = f[sel][np.argmax(p[sel])]
        zs = np.sum((sw - sw.mean()) * np.exp(-2j * np.pi * f0 * t)); zt = np.sum((st - st.mean()) * np.exp(-2j * np.pi * f0 * t))
        depth = min(2 * abs(zs) / len(t) / max(sw.mean(), 1e-6), 2 * abs(zt) / len(t) / max(st.mean(), 1e-6))
        dph = np.degrees(np.angle(zs * np.conj(zt)))
        phases[leg] = np.angle(zs)
        s_alt = (1 - abs(abs(dph) - 180) / 180); s_f = 1.0 if 7 <= f0 <= 15 else max(0.0, 1 - min(abs(f0 - 7), abs(f0 - 15)) / 5)
        total += min(depth, 1.0) * s_alt * s_f
        res[leg] = (round(f0, 1), round(depth, 2), round(dph), round(s_alt * s_f, 2))
    tri = 0.0
    if all(l in phases for l in LEGS):
        A = [phases[l] for l in ("lf", "rm", "lh")]; B = [phases[l] for l in ("rf", "lm", "rh")]
        within = np.mean([np.cos(x - y) for x, y in itertools.combinations(A, 2)] + [np.cos(x - y) for x, y in itertools.combinations(B, 2)])
        between = -np.mean([np.cos(x - y) for x in A for y in B])
        tri = (within + between) / 2
    return total / 6 + 0.5 * max(tri, 0), res, tri


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(783); rows = []
    trials = [{"DNg100": 400}, {"DNg100": 400, "DNa02": 200}, {"DNg100": 400, "DNb08": 200}]
    while len(trials) < ntr:
        trials.append({k: float(rng.choice([0, 0, 100, 200, 300, 400])) for k in DN})
    out = ROOT / "recordings/dn_mix_search.csv"
    for i, amps in enumerate(trials):
        rec, dts = sim(amps); sc, res, tri = score(rec, dts)
        rows.append({**amps, "score": round(sc, 3), "tripod": round(tri, 2), **{f"{l}": str(res[l]) for l in LEGS}})
        print(f"{i:3d} score {sc:.3f} tripod {tri:+.2f} | {amps} | lf {res['lf']} rf {res['rf']}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
