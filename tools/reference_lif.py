"""Reference implementation of the Shiu et al. 2024 (Nature 634:210) whole-brain LIF model, matching
data/raw/model.py (Brian2) exactly in its equations and parameters, written to check our GPU brain:

    dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)     v_0 = v_rst = -52 mV, v_th = -45 mV
    dg/dt = -g / tau                (unless refractory)     t_mbr = 20 ms, tau = 5 ms
    spike: v > v_th -> v = v_rst, g = 0, refractory 2.2 ms (0 ms for Poisson-stimulated neurons)
    synapse: g += 0.275 mV * (sign x synapse count), delay 1.8 ms
    Poisson input: v += 0.275 mV * 250 per event (target_var 'v')
    Brian2 'linear' method = exact integration of the linear ODE over each dt.

Options let the same code run the GPU shader's scheme (`--scheme gpu`: dt 0.5 ms, Euler v, Poisson
into g, refractory for all, delay rounded) to measure what our shortcuts cost.

Benchmark (Shiu et al. Fig. 1): 21 right labellar sugar GRNs Poisson-stimulated at f Hz -> MN9 rate.
usage: .venv/Scripts/python tools/reference_lif.py --scheme shiu --freqs 10 20 ... --trials 5
"""
import argparse, json, pathlib, time
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MN9 = [720575940660219265, 720575940618238523]  # right (Shiu et al.), left = CB0701_L in v783 annotations
# (Shiu et al. list 720575940645521262 for the left MN9, a root ID from an earlier FlyWire release)
SUGAR_R = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
           720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
           720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
           720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
           720575940611875570]


class LIF:
    def __init__(self, scheme="shiu", dt_ms=None, extra_w=None):
        comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
        self.ids = comp.index.to_numpy(np.int64); self.n = len(self.ids)
        self.index = pd.Series(np.arange(self.n), index=self.ids)
        con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                              columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        pre = con.Presynaptic_Index.to_numpy(); post = con.Postsynaptic_Index.to_numpy()
        w = 0.275 * con["Excitatory x Connectivity"].to_numpy(np.float64)
        if extra_w is not None:
            w = extra_w(pre, post, w)
        o = np.argsort(pre, kind="stable")
        self.post, self.w = post[o].astype(np.int64), w[o]
        self.row = np.zeros(self.n + 1, np.int64); np.cumsum(np.bincount(pre, minlength=self.n), out=self.row[1:])
        self.scheme = scheme
        # shiu: Brian2 reference (dt 0.1 ms); gpu: old shader; gpu2: new shader = Shiu semantics at dt 0.5
        self.dt = dt_ms or (0.1 if scheme == "shiu" else 0.5)
        self.delay = int(round(1.8 / self.dt))

    def run(self, stim, t_ms=1000.0, seed=0, record=None):
        """stim: list of (neuron indices, rate Hz). Returns spike counts per neuron."""
        rng = np.random.default_rng(seed)
        n, dt = self.n, self.dt
        v = np.full(n, -52.0); g = np.zeros(n); ref = np.zeros(n)
        rfc = np.full(n, 2.2)
        targets = np.zeros(n, bool)
        for idx, _ in stim:
            targets[idx] = True
        if self.scheme in ("shiu", "gpu2"):
            rfc[targets] = 0.0
        ring = np.zeros((self.delay + 1, n))
        em, es = np.exp(-dt / 20.0), np.exp(-dt / 5.0)
        k = 5.0 / (20.0 - 5.0)                              # tau / (t_mbr - tau)
        counts = np.zeros(n, np.int64)
        steps = int(round(t_ms / dt))
        for s in range(steps):
            slot = s % (self.delay + 1)
            g += ring[slot]; ring[slot] = 0.0
            pois = np.zeros(n)
            for idx, rate in stim:
                pois[idx] += 68.75 * (rng.random(len(idx)) < rate * dt * 1e-3)
            active = ref <= 0
            if self.scheme in ("shiu", "gpu2"):
                # exact solution of the linear system over dt ('linear' method), frozen while refractory
                u = v - (-52.0)
                u_new = u * em + g * k * (em - es)
                v = np.where(active, -52.0 + u_new, v)
                g = np.where(active, g * es, g)
                v += pois                                     # PoissonInput target_var='v'
            else:  # GPU shader scheme: Poisson into g, Euler v, refractory for everyone
                g = g * es + pois
                v = np.where(active, v + dt / 20.0 * (-52.0 - v + g), -52.0)
            ref -= dt
            spk = np.where((v > -45.0) & active)[0] if self.scheme in ("shiu", "gpu2") else np.where((v >= -45.0) & active)[0]
            if len(spk):
                counts[spk] += 1
                v[spk] = -52.0; g[spk] = 0.0; ref[spk] = rfc[spk]
                tgt_slot = (s + self.delay) % (self.delay + 1)
                for i in spk:
                    a, b = self.row[i], self.row[i + 1]
                    np.add.at(ring[tgt_slot], self.post[a:b], self.w[a:b])
        return counts / (t_ms / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="shiu", choices=["shiu", "gpu", "gpu2"])
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--freqs", type=float, nargs="+", default=[20, 40, 60, 80, 100, 150, 200])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = LIF(a.scheme, a.dt)
    sug = m.index.reindex(SUGAR_R).dropna().astype(int).values
    mn9 = m.index.reindex(MN9).dropna().astype(int).values
    res = {}
    for f in a.freqs:
        t0 = time.time()
        r = np.array([m.run([(sug, f)], seed=k)[mn9] for k in range(a.trials)])
        res[f] = (r.mean(0).tolist(), r.std(0).tolist())
        print(f"{a.scheme} dt={m.dt} ms  sugar {f:5.0f} Hz -> MN9 R/L {r.mean(0)[0]:6.1f}/{r.mean(0)[1]:6.1f} Hz "
              f"(sd {r.std(0)[0]:.1f})  [{time.time()-t0:.0f}s]", flush=True)
    out = pathlib.Path(a.out or ROOT / f"recordings/reference_sugar_mn9_{a.scheme}_dt{m.dt}.json")
    out.write_text(json.dumps({"scheme": a.scheme, "dt_ms": m.dt, "trials": a.trials, "mn9_ids": MN9, "rates": res}))


if __name__ == "__main__":
    main()
