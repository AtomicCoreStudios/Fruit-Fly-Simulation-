"""Shared nerve-cord <-> leg model for the MuJoCo body tools (tools/nmf_closed_loop.py, tools/reflex_probe.py).
Network: data/vnc/vnc_rate_net.npz (tools/export_vnc_rate_net.py; Pugliese et al. 2025 rate equations on our BANC
graph). Motor and proprioceptor rules mirror scripts/fly_legs.gd (same basis labels); vectorised.
Needs only numpy/scipy, so it runs in .venv-flygym."""
import json, pathlib
import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
DOFS = ["ThC_pro", "ThC_add", "CTr", "FeRot", "FTi", "TiTa"]
KINDS = ["FeCO_claw", "FeCO_hook", "FeCO_club", "hair_plate", "campaniform"]
R_HALF, TAU_ACT, DTHETA_MN, I_HALF = 30.0, 0.030, 0.4, 500.0
RANGE = {"ThC_pro": [0.6, 0.6], "ThC_add": [0.3, 0.3], "CTr": [0.8, 0.8], "FeRot": [0.4, 0.0],
         "FTi": [1.2, 0.6], "TiTa": [0.5, 0.4]}
DN = {"DNg100": [130297, 135733], "DNb08": [47118, 98130, 115273, 131703], "DNa02": [904, 92992],
      "DNg97": [42812, 78013], "DNp09": [83620, 119032]}


class Cord:
    """Rate network with synaptic filter and delay (as tools/dn_mix_search.py)."""

    def __init__(self, dt=0.0002, delay_ms=1.8, tau_s_ms=5.0):
        net = np.load(ROOT / "data/vnc/vnc_rate_net.npz")
        self.W = sp.csr_matrix((net["W_data"].astype(np.float64), net["W_indices"], net["W_indptr"]))
        self.ids = net["ids"]; self.pos = {int(x): i for i, x in enumerate(self.ids)}; self.n = len(self.ids)
        self.a, self.theta, self.fcap, self.tau = net["a"], net["theta"], float(net["fcap"]), float(net["tau"])
        self.dt, self.tau_s = dt, tau_s_ms
        self.D = int(round(delay_ms / 1000 / dt))
        self.I = np.zeros(self.n)
        self.reset()

    def reset(self):
        self.R = np.zeros(self.n); self.syn = np.zeros(self.n); self.hist = [np.zeros(self.n)] * (self.D + 1)

    def set_drive(self, amps):
        self.I[:] = 0.0
        for k, v in amps.items():
            for x in DN[k]:
                self.I[self.pos[x]] = float(v)

    def step(self, drive_on=True, clamp_idx=None, clamp_val=None):
        self.hist.append(self.R.copy()); Rd = self.hist.pop(0)
        self.syn += self.dt * 1000 * (self.W @ Rd - self.syn) / self.tau_s
        x = (self.I if drive_on else 0.0) + self.syn - self.theta
        self.R += self.dt * (np.maximum(self.fcap * np.tanh((self.a / self.fcap) * x), 0.0) - self.R) / self.tau
        if clamp_idx is not None:
            self.R[clamp_idx] = clamp_val


class Legs:
    """Motor-neuron activation -> joint torque, joint state + ground force -> afferent rates."""

    def __init__(self, cord, k_joint=10.0, sensors=KINDS, cs_muscle=0.0):
        lm = json.load(open(ROOT / "data/leg_motor_map.json"))
        p = cord.pos; self.cord = cord; self.k = k_joint; self.cs_muscle = cs_muscle
        # motor: activation matrix rows = (leg, dof, side) pools, columns = MN slots
        mns = sorted({p[x] for L in lm["legs"].values() for m in L["motor"].values() for s in ("pos", "neg") for x in m[s]})
        self.mn_idx = np.array(mns, int); slot = {x: i for i, x in enumerate(mns)}
        rows, cols, vals = [], [], []
        for li, leg in enumerate(LEGS):
            for di, d in enumerate(DOFS):
                m = lm["legs"][leg]["motor"].get(d)
                if not m:
                    continue
                for si, s in enumerate(("pos", "neg")):
                    for x, w in zip(m[s], m["w_" + s]):
                        rows.append((li * 6 + di) * 2 + si); cols.append(slot[p[x]]); vals.append(float(w))
        self.P = sp.csr_matrix((vals, (rows, cols)), shape=(72, len(mns)))
        self.act = np.zeros(len(mns))
        self.lim_pos = np.array([RANGE[d][0] for _ in LEGS for d in DOFS]); self.lim_neg = np.array([RANGE[d][1] for _ in LEGS for d in DOFS])
        # sensors
        tun = lm.get("feco_tuning", {})
        S = []
        for li, leg in enumerate(LEGS):
            for kind, lst in lm["legs"][leg]["sensory"].items():
                for k, x in enumerate(lst):
                    t = tun.get(str(int(x)), "")
                    S.append((li, KINDS.index(kind), p[int(x)], k / max(len(lst) - 1, 1), (t == "flex") if t else (len(S) % 2 == 0)))
        self.s_leg = np.array([s[0] for s in S]); self.s_kind = np.array([s[1] for s in S]); self.s_idx = np.array([s[2] for s in S])
        self.s_u = np.array([s[3] for s in S]); self.s_flex = np.array([s[4] for s in S], bool)
        # hair plates: joint and limit direction from tools/hair_plate_tuning.py (data/hair_plate_tuning.json);
        # without the file: the old rule (fires at either limit, joint by alternation; positive feedback, 2026-10-08)
        hpt = json.load(open(ROOT / "data/hair_plate_tuning.json")) if (ROOT / "data/hair_plate_tuning.json").exists() else {}
        inv = {i: x for x, i in p.items()}
        self.hp_j = np.array([DOFS.index(hpt[str(inv[i])]["joint"]) if str(inv[i]) in hpt else (0 if fl else 2)
                              for i, fl in zip(self.s_idx, self.s_flex)])
        self.hp_s = np.array([{"pos": 1.0, "neg": -1.0}.get(hpt.get(str(inv[i]), {}).get("limit"), 0.0) for i in self.s_idx])
        self.s_on = np.isin(self.s_kind, [KINDS.index(k) for k in sensors]).astype(float)
        # presynaptic inhibition (Dallmann 2025): afferent row -> inhibitory neurons x synapses
        pre = lm.get("presyn_inhibition", {}); r_, c_, v_ = [], [], []
        where = {int(x): i for i, x in enumerate(self.s_idx)}
        for key, lst in pre.items():
            if p[int(key)] in where:
                for j, sy in lst:
                    r_.append(where[p[int(key)]]); c_.append(p[int(j)]); v_.append(float(sy))
        self.Pre = sp.csr_matrix((v_, (r_, c_)), shape=(len(S), cord.n))

    def torques(self, dt):
        """Activation update and per-DoF torque (uN mm) in 'biological' sign (+ protract/adduct/flex/lower)."""
        r = np.maximum(self.cord.R[self.mn_idx], 0.0)
        self.act += (r / (r + R_HALF) - self.act) * (1 - np.exp(-dt / TAU_ACT))
        A = (self.P @ self.act).reshape(36, 2)
        self.A = A
        eq = DTHETA_MN * (A[:, 0] - A[:, 1])
        lim = np.where(eq >= 0, self.lim_pos, self.lim_neg)
        return np.where(lim > 0, self.k * lim * np.tanh(eq / np.maximum(lim, 1e-3)), 0.0)

    def afferents(self, ang, om, fn, w_body):
        """ang, om: (6 legs, 6 dofs) biological-sign angle re neutral / velocity; fn: (6,) ground force (uN)."""
        L = self.s_leg; th = ang[L, 4]; w = om[L, 4]; sgn = np.where(self.s_flex, 1.0, -1.0)
        claw = 60.0 / (1.0 + np.exp(-sgn * (th - (-0.6 + 1.8 * self.s_u)) / 0.1))
        hook = np.clip(sgn * w * 20.0, 0.0, 120.0)
        club = np.clip(np.abs(w) * 10.0, 0.0, 120.0)
        hj = ang[L, self.hp_j] / np.where(self.hp_j == 0, RANGE["ThC_pro"][0], RANGE["CTr"][0])
        hp_x = np.where(self.hp_s != 0, self.hp_s * hj, np.abs(hj))
        hair = 80.0 / (1.0 + np.exp(-(hp_x - 0.6) / 0.08))
        load = self.A.reshape(6, 12).sum(1) if hasattr(self, "A") else np.zeros(6)
        camp = np.clip(load[L] * 25.0 * self.cs_muscle + 120.0 * fn[L] / w_body, 0.0, 150.0)
        r = np.choose(self.s_kind, [claw, hook, club, hair, camp]) * self.s_on
        return r / (1.0 + (self.Pre @ np.maximum(self.cord.R, 0.0)) / I_HALF)
