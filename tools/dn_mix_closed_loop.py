"""Descending-mix search in closed loop with the MuJoCo body (tools/nmf_closed_loop.py, pad adhesion 10 uN).
Same DN set as tools/dn_mix_search.py (types with documented walking roles, left/right symmetric). Each trial runs
2 s of body physics; score = forward speed over the last 1.5 s (mm/s), minus a fall penalty if the thorax drops below
0.6 mm (passive standing height 0.71-0.76 mm), plus a small credit for stepping (touchdowns in >= 4 legs).
Basis of the result: tuned. Trials run in parallel processes (CPU).
usage: .venv-flygym/Scripts/python tools/dn_mix_closed_loop.py [n_trials=42] [n_parallel=14]
-> recordings/dn_mix_closed_loop.csv"""
import sys, subprocess, pathlib, itertools, csv
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-flygym/Scripts/python")
ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 42
npar = int(sys.argv[2]) if len(sys.argv) > 2 else 14
DNS = ["DNg100", "DNb08", "DNa02", "DNg97", "DNp09"]
rng = np.random.default_rng(1009)
trials = [{}, {"DNg100": 400, "DNb08": 100, "DNa02": 100, "DNg97": 100}, {"DNg100": 800}, {"DNg100": 400}, {"DNp09": 400}]
while len(trials) < ntr:
    trials.append({k: float(rng.choice([0, 0, 200, 400, 800])) for k in DNS})


def score(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows]); sel = t >= 0.5
    x = np.array([float(r["x"]) for r in rows])[sel]; z = np.array([float(r["z"]) for r in rows])[sel]
    dur = t[sel][-1] - t[sel][0]; v = (x[-1] - x[0]) / dur          # body x is anterior at start (straight walking)
    steps = sum(int(np.sum(np.diff(np.array([int(float(r[f"{l}_adh"])) for r in rows])[sel]) > 0) > 2)
                for l in ["lf", "lm", "lh", "rf", "rm", "rh"])
    return v - (2.0 if z.mean() < 0.6 else 0.0) + (0.1 if steps >= 4 else 0.0), v, z.mean(), steps


out = ROOT / "recordings/dn_mix_closed_loop.csv"; res = []
for b in range(0, len(trials), npar):
    procs = []
    for i, amps in enumerate(trials[b:b + npar], start=b):
        a = ",".join(f"{k}:{v:g}" for k, v in amps.items() if v > 0)
        procs.append((i, amps, subprocess.Popen([PY, str(ROOT / "tools/nmf_closed_loop.py"), "--amps", a, "--pad_fmax", "10",
                                                 "--tag", f"_search{i}"], stdout=subprocess.DEVNULL, stderr=open(ROOT / f"recordings/_search{i}.err", "w"))))
    for i, amps, p in procs:
        p.wait()
        f = ROOT / f"recordings/nmf_closed_loop_search{i}.csv"
        err = ROOT / f"recordings/_search{i}.err"
        if not f.exists():
            tail = [l for l in err.read_text(errors="replace").splitlines() if l.strip() and "INFO" not in l][-1:]
            print(f"{i:3d} FAILED ({tail}) | {amps}", flush=True)
            res.append({**{k: amps.get(k, 0) for k in DNS}, "score": float("nan"), "speed_mm_s": float("nan"),
                        "height_mm": float("nan"), "stepping_legs": -1})
            err.unlink(); continue
        sc, v, z, st = score(f); f.unlink(); err.unlink()
        res.append({**{k: amps.get(k, 0) for k in DNS}, "score": round(sc, 3), "speed_mm_s": round(v, 3),
                    "height_mm": round(z, 3), "stepping_legs": st})
        print(f"{i:3d} score {sc:+.3f} speed {v:+.3f} mm/s height {z:.2f} stepping legs {st} | {amps}", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
best = sorted([r for r in res if r["score"] == r["score"]], key=lambda r: -r["score"])[:5]
print("best:", *best, sep="\n  ")
