"""Graded optic lobe beyond FlyVis + graded dendrites for visual projection neurons (VPNs).

Every FlyWire optic-lobe neuron that FlyVis does not drive (LPi, Tm5f, Y, TmY, Li, Dm, Pm, Lawf,
Tlp, ... ~36k) and every visual projection neuron's DENDRITE (LPLC2, LC4, LPLC1, LC6, ... ~8k)
becomes a graded unit with FlyVis dynamics
    tau_i dV_i/dt = -V_i + b_i + sum_j w_ij relu(A_j)
where A_j is either a FlyVis-driven cell's FlyVis activity (in the matching FlyWire neuron) or
another graded unit's V_j. Presynaptic VPNs are NOT used here (their outputs are axonal and spiking;
they stay in the LIF brain).

Parameters (none tuned to the loom result):
  connectivity  FlyWire v783 synapse counts per neuron pair, sign from predicted transmitter
                (ACh +, GABA/Glu -, others 0)                                     basis: measured
  strength      w = sign * n_syn * s_type / kappa
                s_type = median FlyVis-fitted per-synapse strength of the presynaptic type's
                outgoing synapses (global median if the type is not in FlyVis);
                kappa  = median ratio FlyWire / FIB-25 synapse counts over type pairs present in
                both (rescales FlyVis' per-synapse strengths to FlyWire counts)    basis: model-fitted
  tau           FlyVis-fitted tau for types in FlyVis, else the FlyVis median      basis: model-fitted
  b             set so each unit sits at 0 (the rectification knee) under uniform grey light,
                i.e. adapted to mean luminance                                      basis: approximate
VPN dendrite -> spiking FlyWire soma: bias current c * V_dend (mV), c = free parameter, swept.

Writes data/graded_net.npz.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/flyvis"))


def main():
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    ids = comp.index.to_numpy(np.int64)
    n_all = len(ids)
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t",
                    usecols=["root_id", "super_class", "cell_type", "top_nt", "side"], low_memory=False)
    a = a.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    vt = pd.read_csv(ROOT / "data/raw/visual_neuron_types.csv.gz").drop_duplicates("root_id").set_index("root_id").type.reindex(ids)
    ctype = vt.fillna(a.cell_type).fillna("?").to_numpy()
    sc = a.super_class.fillna("").to_numpy()
    nt = a.top_nt.fillna("").to_numpy()

    # FlyVis-driven FlyWire neurons (index -> FlyVis node incl. eye offset)
    fvm = json.loads((ROOT / "data/flyvis_gpu.json").read_text())
    raw = (ROOT / "data/flyvis_gpu.bin").read_bytes()
    o = fvm["offsets"]["out_pairs"]; pairs = np.frombuffer(raw[o[0]:o[0] + o[1]], np.int32).reshape(-1, 2)
    o = fvm["offsets"]["v_grey"]; v_grey = np.frombuffer(raw[o[0]:o[0] + o[1]], np.float32)
    N_fv = fvm["n_nodes"]
    driven = np.full(n_all, -1, np.int64); driven[pairs[:, 0]] = pairs[:, 1]
    is_D = driven >= 0
    is_vpn = sc == "visual_projection"
    is_G = ((sc == "optic") & ~is_D) | is_vpn
    pr_types = {"R1-6", "R7", "R8"}
    is_G &= ~np.isin(ctype, list(pr_types))          # photoreceptors are driven by the eye model
    print(f"FlyVis-driven {is_D.sum()}, graded units {is_G.sum()} (of which VPN dendrites {(is_G & is_vpn).sum()})")

    # ---- FlyVis strengths per presynaptic type, and FlyWire/FIB-25 count ratio kappa
    net = np.load(ROOT / "data/flyvis_net.npz")
    import _env  # noqa: F401
    from flyvis import NetworkView
    fvnet = NetworkView(str(net["model"])).init_network()
    pe = fvnet._param_api().edges
    strength = pe.syn_strength.detach().numpy(); fv_count = pe.syn_count.detach().numpy()
    types = list(net["types"]); ntype = net["ntype"]
    tar = np.repeat(np.arange(len(ntype)), np.diff(net["row"])); src = net["src"]
    # weight was sorted by target in export; syn_strength/syn_count are in connectome edge order
    c = fvnet.connectome
    e_src = np.asarray(c.edges.source_index[:]); e_tar = np.asarray(c.edges.target_index[:])
    st_type = np.array(types)[ntype[e_src]]; tt_type = np.array(types)[ntype[e_tar]]
    s_by_src = pd.Series(strength).groupby(st_type).median()
    s_global = float(np.median(strength))
    # FIB-25 synapses per (src type -> tar type) for one target cell = sum over offsets
    fib = pd.DataFrame({"s": st_type, "t": tt_type, "tar": e_tar, "n": fv_count})
    fib_pair = fib.groupby(["s", "t", "tar"]).n.sum().groupby(["s", "t"]).median()

    con = pd.read_parquet(ROOT / "data/raw/Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    pre = con.Presynaptic_Index.to_numpy(); post = con.Postsynaptic_Index.to_numpy(); nsyn = con.Connectivity.to_numpy()
    fw = pd.DataFrame({"s": ctype[pre], "t": ctype[post], "post": post, "n": nsyn})
    fw = fw[np.isin(fw.s, types) & np.isin(fw.t, types)]
    fw_pair = fw.groupby(["s", "t", "post"]).n.sum().groupby(["s", "t"]).median()
    both = fib_pair.index.intersection(fw_pair.index)
    ratio = (fw_pair[both] / fib_pair[both]).replace([np.inf, 0], np.nan).dropna()
    kappa = float(np.median(ratio))
    print(f"kappa (FlyWire / FIB-25 synapses per connection, median over {len(ratio)} type pairs) = {kappa:.2f}; "
          f"FlyVis per-synapse strength: global median {s_global:.4f}")
    # FlyVis' fitted strengths fall with the target's total input: strength ~ (total input)^beta
    tot_fv = fib.groupby("tar").n.sum(); fib["tot"] = fib.tar.map(tot_fv)
    pp = fib.assign(st=strength).groupby(["s", "t"]).agg(st=("st", "first"), tot=("tot", "median"))
    pp = pp[pp.st > 1e-6]
    beta = float(np.polyfit(np.log10(pp.tot), np.log10(pp.st), 1)[0])
    T0 = float(np.median(tot_fv)) / kappa * 1.0
    print(f"FlyVis in-degree rule: strength ~ (total input synapses)^{beta:.2f}, reference {T0:.0f} synapses")

    # ---- graded connectivity: post in G, pre in D or (G minus VPN)
    pre_ok = is_D | (is_G & ~is_vpn)
    m = is_G[post] & pre_ok[pre]
    sign = np.where(np.isin(nt, ["acetylcholine"]), 1.0, np.where(np.isin(nt, ["gaba", "glutamate"]), -1.0, 0.0))
    s_pre = pd.Series(ctype).map(s_by_src).fillna(s_global).to_numpy()
    tot_in = np.bincount(post, weights=nsyn, minlength=n_all)        # all FlyWire input synapses
    fan = (np.maximum(tot_in[post[m]], 1) * kappa / T0) ** beta    # in FIB-25-equivalent units
    w = (sign[pre[m]] * nsyn[m] * s_pre[pre[m]] / kappa * fan).astype(np.float32)
    keep = w != 0
    gpre, gpost, w = pre[m][keep], post[m][keep], w[keep]
    # compact index spaces: A = [D cells..., graded cells...]
    D_idx = np.where(is_D)[0]; G_idx = np.where(is_G)[0]
    a_of = np.full(n_all, -1, np.int64)
    a_of[D_idx] = np.arange(len(D_idx)); a_of[G_idx] = len(D_idx) + np.arange(len(G_idx))
    g_of = np.full(n_all, -1, np.int64); g_of[G_idx] = np.arange(len(G_idx))
    col = a_of[gpre]; rowi = g_of[gpost]
    order = np.argsort(rowi, kind="stable"); col, rowi, w = col[order], rowi[order], w[order]
    row_ptr = np.zeros(len(G_idx) + 1, np.int64); np.cumsum(np.bincount(rowi, minlength=len(G_idx)), out=row_ptr[1:])
    # Stability: FlyVis' own fitted network has max real eigenvalue 0.89.
    # One global scale alpha brings the recurrent graded block (G -> G) to the same value.
    import scipy.sparse as sp, scipy.sparse.linalg as sl
    nD = len(D_idx); nG = len(G_idx)
    Wgg = sp.csr_matrix((w.astype(np.float64), col, row_ptr), shape=(nG, nD + nG))[:, nD:]
    # continuous-time stability is set by the largest REAL part (not the magnitude: large negative
    # eigenvalues are mutual-inhibition loops and are stable); FlyVis' fitted W has max Re = 0.89
    rho = float(np.max(sl.eigs(Wgg, k=8, which="LR", return_eigenvectors=False, maxiter=5000).real))
    RHO_FLYVIS = 0.89
    alpha = min(1.0, RHO_FLYVIS / rho)
    w = (w * alpha).astype(np.float32)
    print(f"graded G->G max real eigenvalue {rho:.2f} before scaling -> global scale alpha = {alpha:.3f}")
    # tau
    tau_by_type = dict(zip(types, [float(np.median(net["tau"][ntype == k])) for k in range(len(types))]))
    tau_med = float(np.median(net["tau"]))
    tau = np.array([tau_by_type.get(t, tau_med) for t in ctype[G_idx]], np.float32)
    # grey-adapted bias: drive from D at grey; G units at 0 => b = -sum_j w_ij relu(A_j_grey)
    A_grey = np.zeros(len(D_idx) + len(G_idx), np.float32)
    A_grey[:len(D_idx)] = np.maximum(v_grey[driven[D_idx] % N_fv], 0)
    drive_grey = np.bincount(rowi, weights=w * A_grey[col], minlength=len(G_idx))
    bias = (-drive_grey).astype(np.float32)
    print(f"graded edges {len(w)} ({nsyn[m][keep].sum()} synapses); excitatory {np.mean(w > 0):.0%}; "
          f"tau median {np.median(tau)*1000:.0f} ms")
    # LIF edges to silence (now carried by the graded model): pre in D or G\VPN, post in D or G
    np.savez_compressed(ROOT / "data/graded_net.npz", D_idx=D_idx, D_node=driven[D_idx], G_idx=G_idx,
                        G_type=ctype[G_idx], G_vpn=is_vpn[G_idx], row_ptr=row_ptr, col=col.astype(np.int32),
                        w=w, tau=tau, bias=bias, kappa=kappa, s_global=s_global, alpha=alpha, beta=beta, rho=rho,
                        silence_pre=pre_ok, silence_post=(is_D | is_G))
    print("wrote data/graded_net.npz")


if __name__ == "__main__":
    main()
