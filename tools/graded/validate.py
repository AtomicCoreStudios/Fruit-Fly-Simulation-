"""Offline validation of the graded optic lobe + VPN dendrites against published physiology.

Stimuli are rendered on the RIGHT eye's real facet directions (FlyVis lattice as placed by
tools/flyvis/map_eye_lattice.py), run through FlyVis (torch, exact) and then the graded FlyWire
network (data/graded_net.npz, same equations as the GPU shader). Left eye held at grey.

Tests (Klapoetke et al. 2017 Nature 551:237; Ache et al. 2019 Curr Biol 29:1073):
  LPLC2 should respond to a dark loom and an outward-moving cross, and NOT (or much less) to a
  receding disc, an inward cross, wide-field translation, or luminance-matched motion-free dimming.
  LC4 should track angular velocity, LPLC2 angular size (r/v sweep).
usage: .venv/Scripts/python tools/graded/validate.py [--plot screenshots/graded_validation.png]
"""
import argparse, json, pathlib, sys
import numpy as np, scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/flyvis"))
import _env  # noqa: E402,F401
import torch  # noqa: E402
from flyvis import NetworkView  # noqa: E402

DT = 1 / 200
BG = 0.5
PRE = 200   # 1 s lead-in


class Sim:
    def __init__(self):
        self.g = np.load(ROOT / "data/graded_net.npz", allow_pickle=True)
        net = np.load(ROOT / "data/flyvis_net.npz")
        self.N_fv = len(net["u"])
        self.fv = NetworkView(str(net["model"])).init_network(); self.fv.eval()
        mp = json.loads((ROOT / "data/flyvis_map.json").read_text())["eyes"]["right"]
        dirs = np.array(json.loads((ROOT / "data/eye_drive.json").read_text())["facet_dir"])
        d = dirs[mp["hexal_facet"]]
        self.az = np.degrees(np.arctan2(-d[:, 2], d[:, 0])); self.el = np.degrees(np.arcsin(d[:, 1]))
        g = self.g
        nG = len(g["G_idx"]); nA = len(g["D_idx"]) + nG
        self.W = sp.csr_matrix((g["w"], g["col"], g["row_ptr"]), shape=(nG, nA))
        self.tau = np.maximum(g["tau"], DT); self.b = g["bias"]
        with torch.no_grad():
            self.st = self.fv.steady_state(1.0, DT, 1)
        self.v_grey = self.st.nodes.activity.numpy().reshape(-1)
        self.types = g["G_type"]
        # right-hemisphere graded units per type (side from FlyWire annotation)
        import pandas as pd
        a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t",
                        usecols=["root_id", "side"]).drop_duplicates("root_id").set_index("root_id")
        comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
        self.side = a.side.reindex(comp.index).fillna("").to_numpy()[g["G_idx"]]

    def run(self, movie):
        """movie: (T, 721) luminance on the right eye. Returns graded V (T, nG)."""
        with torch.no_grad():
            fv = self.fv.simulate(torch.tensor(movie, dtype=torch.float32)[None, :, None], DT,
                                  initial_state=self.st).numpy()[0]           # (T, N_fv) right eye
        D_node = self.g["D_node"]; right = D_node < self.N_fv
        nD = len(D_node); nG = len(self.b)
        A = np.zeros(nD + nG, np.float32)
        A[:nD] = np.maximum(self.v_grey[D_node % self.N_fv], 0)          # left eye stays at grey
        V = np.zeros(nG, np.float32); out = np.empty((len(movie), nG), np.float32)
        for t in range(len(movie)):
            A[:nD][right] = np.maximum(fv[t, D_node[right]], 0)
            A[nD:] = np.maximum(V, 0)
            V = V + DT / self.tau * (-V + self.b + self.W @ A)
            out[t] = V
        return out

    def pop(self, V, t, side="right"):
        m = (self.types == t) & (self.side == side)
        r = np.maximum(V[:, m], 0)
        return r.mean(1), r.max(1)


# ---------------------------------------------------------------- stimuli (angles in degrees)
def ang_dist(az, el, az0, el0):
    a, e, a0, e0 = map(np.radians, (az, el, az0, el0))
    return np.degrees(np.arccos(np.clip(np.sin(e) * np.sin(e0) + np.cos(e) * np.cos(e0) * np.cos(a - a0), -1, 1)))


def soft(x, w=2.0):
    return 1 / (1 + np.exp(-x / w))


def loom_theta(rv_ms, t, t_c, th0=5.0, th_max=90.0):
    """half-angle (deg) of an object of radius/speed rv approaching, collision at t_c"""
    tt = np.maximum(t_c - t, 1e-4)
    return np.clip(np.degrees(np.arctan(rv_ms / 1000 / tt)), th0, th_max)


def stimuli(S, centre=(-90.0, 10.0), rv=40.0):
    T = int(1.6 / DT); t = np.arange(T) * DT; t_c = 1.2
    r = ang_dist(S.az, S.el, *centre)
    th = loom_theta(rv, t, t_c)
    loom = BG - BG * soft(th[:, None] - r[None, :])                         # dark disc
    recede = loom[::-1].copy()
    # luminance-matched dimming: final disc area darkens with the loom's mean luminance time course
    area = r < th.max()
    frac = np.clip((th[:, None] ** 2) / (th.max() ** 2), 0, 1)              # fraction of final area dark
    dim = np.full_like(loom, BG); dim[:, area] = (BG * (1 - frac))[:, 0][:, None]
    # wide-field translation: square-wave grating, 30 deg period, 60 deg/s, front-to-back
    lam, spd = 30.0, 60.0
    trans = BG + 0.5 * BG * np.sign(np.sin(2 * np.pi * (S.az[None, :] + spd * t[:, None]) / lam))
    # cross of 4 bars (10 deg wide) whose ends move outward (or inward) from the centre
    daz = (S.az - centre[0]) * np.cos(np.radians(centre[1])); de = S.el - centre[1]
    on_bar = (np.abs(daz) < 5) | (np.abs(de) < 5)
    ext = np.minimum(np.maximum(np.abs(daz), np.abs(de)), 60)
    L = th * 2 / 3                                                             # arm length grows like the loom
    out_cross = np.where(on_bar[None] & (ext[None] < L[:, None]), 0.0, BG)
    in_cross = out_cross[::-1].copy()
    st = {"loom": loom, "recede": recede, "dimming": dim, "translation": trans,
          "outward cross": out_cross, "inward cross": in_cross}
    # 1 s static lead-in of each stimulus' first frame (avoids an onset flash on a grey-adapted
    # eye, as in the physiology protocols); responses are measured after the lead-in only
    st = {k: np.concatenate([np.repeat(v[:1], PRE, 0), v]) for k, v in st.items()}
    return st, np.concatenate([np.arange(-PRE, 0) * DT, t]), np.concatenate([np.full(PRE, th[0]), th])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plot", default="screenshots/graded_validation.png")
    a = ap.parse_args()
    S = Sim()
    # stability check under grey
    Vg = S.run(np.full((200, 721), BG, np.float32))
    print(f"grey 1 s: max |V| graded units {np.abs(Vg[-1]).max():.3f}")
    stims, t, th = stimuli(S)
    TYPES = ["LPLC2", "LC4", "LPLC1", "LC6", "LPi34", "LPi43", "LPi2b", "Tm5f", "Y3", "LLPC1"]
    present = [x for x in TYPES if ((S.types == x) & (S.side == "right")).any()]
    lpi = sorted({x for x in S.types if str(x).startswith("LPi")})
    print("LPi types present:", lpi)
    res = {}
    for name, mov in stims.items():
        V = S.run(mov.astype(np.float32))
        res[name] = {x: tuple(r[PRE:] for r in S.pop(V, x)) for x in present}
        print(f"{name:14s} " + "  ".join(f"{x} {res[name][x][0].max():.3f}" for x in present))
    print("\nSelectivity (peak population response relative to loom):")
    for x in ["LPLC2", "LC4"]:
        base = res["loom"][x][0].max()
        print(f"  {x}: " + ", ".join(f"{k} {res[k][x][0].max() / max(base, 1e-9):.2f}" for k in stims))
    # r/v sweep (Ache et al. 2019)
    print("\nr/v sweep: time (ms before collision) of peak, and half-angle at peak")
    sweep = {}
    for rv in (10, 20, 40, 80):
        st_, t_, th_ = stimuli(S, rv=rv)
        V = S.run(st_["loom"].astype(np.float32))
        for x in ["LPLC2", "LC4"]:
            m = S.pop(V, x)[0][PRE:]; k = int(np.argmax(m))
            sweep[(rv, x)] = (m, th_[PRE:])
            print(f"  r/v {rv:3d} ms  {x:5s}: peak {m.max():.3f} at {1200 - t_[PRE:][k]*1000:+.0f} ms before collision, "
                  f"half-angle {th_[PRE:][k]:.0f} deg")
    np.savez_compressed(ROOT / "recordings/graded_validation.npz", t=t[PRE:], th=th[PRE:],
                        **{f"{k}|{x}": v[0] for k, d in res.items() for x, v in d.items()})
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for k, c in zip(stims, ["k", "#999", "#8c564b", "#2ca02c", "#d62728", "#1f77b4"]):
            ax[0].plot(t[PRE:], res[k]["LPLC2"][0], color=c, label=k)
            ax[1].plot(t[PRE:], res[k]["LC4"][0], color=c, label=k)
        ax[0].set_title("LPLC2 (right, graded dendrites, population mean)"); ax[1].set_title("LC4")
        for rv, c in zip((10, 20, 40, 80), ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]):
            m, th_ = sweep[(rv, "LPLC2")]; ax[2].plot(th_, m, color=c, label=f"LPLC2 r/v {rv}")
            m, th_ = sweep[(rv, "LC4")]; ax[2].plot(th_, m, color=c, ls="--", label=f"LC4 r/v {rv}")
        ax[2].set_xlabel("loom half-angle (deg)"); ax[2].set_title("r/v sweep vs angular size")
        for x in ax[:2]:
            x.set_xlabel("time (s); collision at 1.2 s"); x.axvline(1.2, color="0.7", ls=":")
        ax[0].legend(fontsize=8); ax[2].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(ROOT / a.plot, dpi=110); print("wrote", a.plot)


if __name__ == "__main__":
    main()
