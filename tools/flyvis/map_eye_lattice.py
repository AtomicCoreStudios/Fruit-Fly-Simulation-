"""Place the FlyVis hex lattice (721 columns) onto each real eye and write data/flyvis_map.json.

Orientation: FlyVis T4/T5 preferred directions measured in export_flyvis.py are anchored to biology
(Maisak et al. 2013 Nature): T4a/T5a = front-to-back, T4b/T5b = back-to-front, T4c/T5c = upward,
T4d/T5d = downward. Among the 12 symmetries of the hex lattice we pick, per eye, the map
(u,v) -> Zhao/Codex (p,q) whose physical directions (from the measured facet axes) best match.
basis: approximate (a model lattice laid onto a measured one; ~721 of ~800 columns covered).

Each FlyVis column then reads the facet at its (p,q) (nearest facet if that column has none), and
drives the FlyWire neurons of the Codex column at that (p,q).
"""
import json, pathlib, itertools
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXT = 15
BIO = {"a": "posterior", "b": "anterior", "c": "dorsal", "d": "ventral"}  # Maisak et al. 2013


def hex_group():
    R = np.array([[0, -1], [1, 1]]); F = np.array([[0, 1], [1, 0]])
    out, M = [], np.eye(2, dtype=int)
    for _ in range(6):
        out += [M.copy(), M @ F]
        M = R @ M
    S = np.array([[1, 0], [0, -1]])  # converts a (1,-1)-diagonal lattice to a (1,1)-diagonal one
    return out + [S @ G for G in out]


def main():
    net = np.load(ROOT / "data/flyvis_net.npz")
    om = json.loads((ROOT / "data/ommatidia.json").read_text())["facets"]
    drive = json.loads((ROOT / "data/eye_drive.json").read_text())
    dirs = np.array(drive["facet_dir"])         # Godot head-local: +X anterior, +Y dorsal, -Z left
    hex_u, hex_v = net["hex_u"], net["hex_v"]
    pd = dict(zip(net["pd_types"], net["pd_deg"]))
    # FlyVis pixel frame -> axial step: x = 1.5 v, y = -sqrt(3)(u + v/2)
    def px_to_uv(dx, dy):
        dv = dx / 1.5
        du = -dy / np.sqrt(3) - dv / 2
        return np.array([du, dv])
    out = {"note": __doc__.strip().splitlines()[0], "eyes": {}}
    # Zhao lattice neighbours: which diagonal is a true neighbour (~4.8 deg)?
    for side in ("right", "left"):
        idx = [i for i, f in enumerate(om) if f["side"] == side and f["pq"]]
        pq = {tuple(om[i]["pq"]): i for i in idx}
        az = np.degrees(np.arctan2(-dirs[:, 2], dirs[:, 0])); el = np.degrees(np.arcsin(dirs[:, 1]))
        ant_sign = 1.0 if side == "right" else -1.0      # anterior = toward az 0
        # local Jacobian (dp, dq) -> (d_anterior, d_dorsal) in degrees, fitted from lattice steps
        A, B = [], []
        for (p, q), i in pq.items():
            for dp, dq in ((1, 0), (0, 1), (1, 1), (1, -1)):
                j = pq.get((p + dp, q + dq))
                if j is not None:
                    A.append((dp, dq)); B.append((ant_sign * (az[j] - az[i]), el[j] - el[i]))
        A, B = np.array(A, float), np.array(B)
        diag = {d: np.median(np.hypot(*B[(A == d).all(1)].T)) for d in ((1, 1), (1, -1))}
        J = np.linalg.lstsq(A[(A != (1, -1) if diag[(1, 1)] < diag[(1, -1)] else A != (1, 1)).any(1)],
                            B[(A != (1, -1) if diag[(1, 1)] < diag[(1, -1)] else A != (1, 1)).any(1)], rcond=None)[0].T
        zhao_diag = (1, 1) if diag[(1, 1)] < diag[(1, -1)] else (1, -1)
        best = None
        for M in hex_group():
            # FlyVis neighbours (1,0),(0,1),(1,-1) must map to Zhao neighbours
            nbrs = {tuple(M @ np.array(n)) for n in ((1, 0), (0, 1), (1, -1))}
            ok = all(n in {(1, 0), (0, 1), zhao_diag, (-1, 0), (0, -1), tuple(-np.array(zhao_diag))} for n in nbrs)
            if not ok:
                continue
            score, detail = 0.0, {}
            for t, deg in pd.items():
                want = {"posterior": (-1, 0), "anterior": (1, 0), "dorsal": (0, 1), "ventral": (0, -1)}[BIO[t[-1]]]
                uv = px_to_uv(np.cos(np.radians(deg)), np.sin(np.radians(deg)))
                phys = J @ (M @ uv)
                cosang = phys @ np.array(want) / np.linalg.norm(phys)
                score += cosang; detail[t] = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
            if best is None or score > best[0]:
                best = (score, M, detail)
        score, M, detail = best
        # centre: maximise facets with neurons inside the FlyVis disc
        P = np.array(list(pq.keys()))
        Minv = np.round(np.linalg.inv(M)).astype(int)
        best_c = None
        for p0 in range(-6, 7):
            for q0 in range(-6, 7):
                uv = (Minv @ (P - [p0, q0]).T).T
                inside = (np.abs(uv[:, 0]) <= EXT) & (np.abs(uv[:, 1]) <= EXT) & (np.abs(uv.sum(1)) <= EXT)
                n_in = int(sum(bool(om[pq[tuple(x)]]["neurons"]) for x in P[inside]))
                if best_c is None or n_in > best_c[0]:
                    best_c = (n_in, (p0, q0))
        n_in, (p0, q0) = best_c
        hexal_pq, hexal_facet, exact = [], [], 0
        for u, v in zip(hex_u, hex_v):
            p, q = (M @ np.array([u, v])) + [p0, q0]
            hexal_pq.append([int(p), int(q)])
            if (p, q) in pq:
                hexal_facet.append(pq[(p, q)]); exact += 1
            else:   # nearest facet in hex distance (eye edge / lattice gaps)
                d = [abs(p - a) + abs(q - b) for a, b in P]
                hexal_facet.append(pq[tuple(P[int(np.argmin(d))])])
        out["eyes"][side] = {"M_uv_to_pq": M.tolist(), "centre_pq": [p0, q0], "hexal_pq": hexal_pq,
                             "hexal_facet": hexal_facet, "exact": exact,
                             "pd_error_deg": detail}
        print(f"{side}: map {M.tolist()} centre {(p0, q0)}, {exact}/{len(hex_u)} FlyVis columns on a real facet, "
              f"{n_in} facets with FlyWire neurons covered; T4/T5 direction error vs biology (deg): {detail}")
    (ROOT / "data/flyvis_map.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
