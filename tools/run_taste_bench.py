"""Shiu et al. 2024 Fig. 1 benchmark run inside the embodied Godot brain: stimulate a set of labellar
GRNs at f Hz (no other input), record MN9 (CB0701) rates every 100 ms, compare with the exact
reference model (tools/reference_lif.py, recordings/reference_sugar_mn9_shiu_dt0.1.json).
usage: run_taste_bench.py [--set shiu|sugar] [--freqs 20 40 ...]"""
import argparse, json, pathlib, subprocess
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
GODOT = r"F:\GODOT4.6\Godot_v4.6-stable_win64_console.exe"


def run(f, sset, frames=420, extra=()):
    csv = f"recordings/taste_bench_{sset}_{f:g}.csv"
    subprocess.run([GODOT, "--path", str(ROOT), "--disable-vsync", "--fixed-fps", "30", "--quit-after", str(frames * 4), "--", "--tethered", "--vision=none",
                    f"--frames={frames}", f"--taste_bench={f}", f"--taste_set={sset}", f"--record={csv}", *extra],
                   capture_output=True, timeout=900)
    d = pd.read_csv(ROOT / csv)
    d = d[d.t_ms > 1000]
    return d["type:MN9_R"].mean(), d["type:MN9_L"].mean(), d.proboscis.mean() if "proboscis" in d else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="shiu")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Godot user args, e.g. --phys_ablate=adapt")
    ap.add_argument("--freqs", type=float, nargs="+", default=[20, 40, 60, 80, 100, 150, 200])
    a = ap.parse_args()
    ref = {}
    rp = ROOT / "recordings/reference_sugar_mn9_shiu_dt0.1.json"
    if rp.exists():
        ref = {float(k): v[0] for k, v in json.loads(rp.read_text())["rates"].items()}
    rows = []
    for f in a.freqs:
        r, l, per = run(f, a.set, extra=a.extra)
        rr = ref.get(f, [float("nan")] * 2)
        rows.append((f, r, l, rr[0], rr[1], per))
        print(f"{a.set} {f:5.0f} Hz -> MN9 R/L {r:6.1f}/{l:6.1f} Hz   exact reference R/L {rr[0]:6.1f}/{rr[1]:6.1f}   PER {per:.2f}", flush=True)
    pd.DataFrame(rows, columns=["freq", "mn9_R", "mn9_L", "ref_R", "ref_L", "per"]).to_csv(
        ROOT / f"recordings/taste_bench_{a.set}.csv", index=False)


if __name__ == "__main__":
    main()
