"""Phase 5 loom test: tethered fly, dark sphere (r = 0.7 cm, 7 cm/s, from 9 cm) launched at a fixed
bearing. One Godot run per condition; per-population rates recorded every 100 ms of emulated time.
Reports each visual-pathway population's rate in the approach window vs the pre-launch baseline.

usage: .venv/Scripts/python tools/run_loom_test.py [--bias 0 5 6 6.5] [--angle 90] [--control]
"""
import argparse, pathlib, re, subprocess
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
GODOT = r"F:\GODOT4.6\Godot_v4.6-stable_win64_console.exe"
KEY = ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4a", "T4b", "T4c", "T4d",
       "T5a", "T5b", "T5c", "T5d", "LC4", "LPLC1", "LPLC2", "LC6", "DNp01", "DNp02", "DNp11"]


def d_last_launch_guess(out):
    # control run: use the frame the threat would have launched (~4 s), from the log timestamps
    return 4000.0


def run(bias, angle, threat=True, frames=480, launch=240, optic="lif", gain=60.0, vpn_c=19.0):
    tag = {"lif": f"loom_b{bias}_a{angle}", "flyvis": f"loom_flyvis_g{gain:g}_a{angle}",
           "graded": f"loom_graded_c{vpn_c:g}_a{angle}"}[optic] + ("" if threat else "_ctrl")
    csv = f"recordings/{tag}.csv"
    args = [GODOT, "--path", str(ROOT), "--quit-after", str(frames * 4), "--", "--tethered", "--log",
            f"--frames={frames}", f"--record={csv}", f"--columnar_bias={bias}",
            f"--threat_at={launch if threat else 10**9}", f"--threat_angle={angle}",
            f"--shot=screenshots/{tag}.png", f"--optic_lobe={'lif' if optic == 'lif' else 'flyvis'}",
            f"--flyvis_gain={gain}", f"--graded={1 if optic == 'graded' else 0}", f"--vpn_c={vpn_c}",
            "--probe=data/loom_probe.json", f"--probe_out=recordings/{tag}_probe.csv"]
    out = subprocess.run(args, capture_output=True, text=True, timeout=600).stdout
    (ROOT / f"recordings/{tag}.log").write_text(out)
    m = re.search(r"THREAT launched at t=([\d.]+)s", out)
    if threat and not m:
        raise RuntimeError(f"{tag}: threat never launched - see recordings/{tag}.log")
    t0 = float(m.group(1)) * 1000 if m else d_last_launch_guess(out)
    esc = re.findall(r"ESCAPE \(giant fibre\) at t=([\d.]+)s", out)
    d = pd.read_csv(ROOT / csv)
    base = d[(d.t_ms > t0 - 2000) & (d.t_ms <= t0)]
    loom = d[(d.t_ms > t0) & (d.t_ms <= t0 + 1300)]
    rows = []
    for t in KEY:
        for sd in "LR":
            c = f"type:{t}_{sd}"
            if c in d:
                rows.append((t, sd, base[c].mean(), loom[c].mean(), loom[c].max()))
    return tag, t0, esc, pd.DataFrame(rows, columns=["type", "side", "base_hz", "loom_mean_hz", "loom_peak_hz"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, nargs="+", default=[0.0])
    ap.add_argument("--angle", type=float, default=90)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--optic", default="lif", choices=["lif", "flyvis", "graded"])
    ap.add_argument("--vpn_c", type=float, nargs="+", default=[19.0])
    ap.add_argument("--gain", type=float, nargs="+", default=[60.0])
    a = ap.parse_args()
    for b in (a.bias if a.optic == "lif" else [0.0]):
      for g in (a.gain if a.optic != "lif" else [60.0]):
       for c in (a.vpn_c if a.optic == "graded" else [19.0]):
        for thr in ([True, False] if a.control else [True]):
            tag, t0, esc, df = run(b, int(a.angle), thr, optic=a.optic, gain=g, vpn_c=c)
            print(f"\n=== {tag}: launch {t0/1000:.2f}s, giant-fibre escapes at {esc or 'none'}")
            w = df.pivot(index="type", columns="side", values=["base_hz", "loom_mean_hz", "loom_peak_hz"]).reindex(KEY)
            print(w.round(1).to_string())
            df.to_csv(ROOT / f"recordings/{tag}_summary.csv", index=False)
