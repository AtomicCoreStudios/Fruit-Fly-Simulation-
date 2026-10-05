"""Stepping analysis of leg joint angles from a probe CSV (columns joint_<leg>_<dof>, written when the
neuromuscular legs are on): per leg the dominant frequency and rhythmicity of each joint, and the phase of each
leg's femur-tibia (FTi) rhythm relative to the left front leg. Tripod gait: L1, R2, L3 in phase, anti-phase
(~180 deg) to R1, L2, R3. usage: analyze_gait.py <probe csv> [t_start_ms]"""
import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
d = d[d.t_ms > t0]; t = d.t_ms.values / 1000; fs = 1 / np.median(np.diff(t))
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
def spec(x):
    x = (x - x.mean()) * np.hanning(len(x)); p = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    band = (f >= 1) & (f <= 40); sel = (f >= 2) & (f <= 30); fp = f[sel][np.argmax(p[sel])]
    return fp, p[(f >= fp - 1) & (f <= fp + 1)].sum() / max(p[band].sum(), 1e-12)
rows = []
for leg in LEGS:
    for dof in ["ThC_pro", "CTr", "FTi", "TiTa"]:
        c = f"joint_{leg}_{dof}"
        if c in d:
            x = d[c].values; fp, sc = spec(x)
            rows.append((leg, dof, x.std(), fp, sc))
r = pd.DataFrame(rows, columns=["leg", "dof", "amp_rad_sd", "peak_hz", "rhythmicity"])
print(f"sampling {fs:.0f} Hz"); print(r.round(3).pivot(index="leg", columns="dof", values=["amp_rad_sd", "peak_hz", "rhythmicity"]).to_string())
ref = d["joint_lf_FTi"].values - d["joint_lf_FTi"].mean()
fp, _ = spec(d["joint_lf_FTi"].values)
print(f"\nphase of FTi vs left front at {fp:.1f} Hz (deg):")
for leg in LEGS:
    y = d[f"joint_{leg}_FTi"].values - d[f"joint_{leg}_FTi"].mean()
    ph = np.angle(np.sum(y * np.exp(-2j * np.pi * fp * t)) / np.sum(ref * np.exp(-2j * np.pi * fp * t)), deg=True)
    print(f"  {leg}: {ph:6.0f}")
