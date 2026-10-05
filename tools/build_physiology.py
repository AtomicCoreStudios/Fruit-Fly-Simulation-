"""Per-neuron intrinsic physiology for the LIF brain: spontaneous activity and spike-frequency
adaptation, from data/physiology_rules.json (literature table) -> data/physiology.bin.

Neuron model (identical to shaders/lif.glsl, exact integration at dt = 0.5 ms):
  dv/dt = (v_rest - v + g + bias + mu + x - w) / tau_m       (x: OU noise, sigma, tau_noise)
  dw/dt = -w / tau_w ;  on spike: w += b                      (adaptation current, mV)
Calibration (single neuron, no synaptic input):
  mu(rate): resting offset that gives the target spontaneous rate under the noise
  b(index): adaptation increment that makes a step input, which first drives the neuron to
            100 Hz, settle to adapt_index x 100 Hz
Output per neuron (model order: FlyWire neurons, then nerve-cord neurons): vec4(mu, sigma, b, tau_w);
neurons excluded by the rules get (0, 0, 0, 1) i.e. unchanged Shiu et al. behaviour.
"""
import json, os, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DT, TM, TS, VR, VTH, TREF = 0.5, 20.0, 5.0, -52.0, -45.0, 2.2


def simulate(mu, sigma, b, tau_w, tau_n, t_ms, drive=0.0, n=400, seed=0):
    """vectorised single-neuron simulations; mu/b arrays of length n. Returns spike counts per 50-ms bin."""
    rng = np.random.default_rng(seed)
    v = np.full(n, VR); x = np.zeros(n); w = np.zeros(n); ref = np.zeros(n)
    em = np.exp(-DT / TM); en = np.exp(-DT / tau_n); ew = np.exp(-DT / tau_w)
    steps = int(t_ms / DT); bins = np.zeros((steps // 100 + 1, n))
    for s in range(steps):
        x = x * en + sigma * np.sqrt(1 - en * en) * rng.standard_normal(n)
        w = w * ew
        inp = mu + x - w + drive
        act = ref <= 0
        v = np.where(act, VR + inp + (v - VR - inp) * em, v)
        ref -= DT
        spk = act & (v > VTH)
        v[spk] = VR; ref[spk] = TREF; w[spk] += b[spk] if np.ndim(b) else b
        bins[s // 100] += spk
    return bins


def calibrate(sigma, tau_n, tau_w):
    mus = np.linspace(-6, 14, 400)
    rate = simulate(mus, sigma, 0.0, tau_w, tau_n, 20000).sum(0) / 20.0
    # drive giving ~100 Hz onset without adaptation, then b grid
    drives = np.linspace(5, 60, 400)
    r0 = simulate(np.zeros(400), sigma, 0.0, tau_w, tau_n, 2000, drive=drives).sum(0) / 2.0
    d100 = float(np.interp(100.0, r0, drives))
    bs = np.linspace(0, 4, 400)
    bins = simulate(np.zeros(400), sigma, bs, tau_w, tau_n, 3000, drive=d100)
    onset = bins[:2].sum(0) / 0.1                      # first 100 ms
    steady = bins[-20:].sum(0) / 1.0                   # last second
    index = steady / np.maximum(onset, 1)
    return mus, rate, bs, index, d100


def _vnc_sizes(n_model):
    """Relative neuron size for the VNC rate units (Pugliese et al.: a /= size, theta *= size): BANC surface area
    from their table where measured, else predicted from skeleton cable length (BANC Dataverse skeletons;
    log area = 1.037 log cable + 1.179, r = 0.947 on 4286 neurons), else exp(4.64) x synapses^0.60 um2 (fit log area vs log synapse count over
    3151 measured VNC neurons, r = 0.90), divided by the median area of their measured subnetwork."""
    size = np.ones(n_model)
    vp = ROOT / "data/vnc/vnc_neurons.csv"; pp = ROOT / "data/raw/pugliese/wTable_20260217_fullData_consistentColumns.csv"
    if not vp.exists():
        return size
    v = pd.read_csv(vp); e = pd.read_parquet(ROOT / "data/vnc/vnc_edges.parquet")
    cnt = e.groupby("Postsynaptic_Index").Connectivity.sum().add(e.groupby("Presynaptic_Index").Connectivity.sum(), fill_value=0)
    sa = {}
    if pp.exists():
        pt = pd.read_csv(pp); sa = dict(zip(pt.pt_root_id.astype("int64"), pt.surf_area_um2))
    k, c = 0.60, 4.64                                       # log(area um2) = 0.60 log(synapses) + 4.64
    area = np.array([sa.get(b_, np.nan) for b_ in v.bid], float)
    est = np.exp(c + k * np.log(np.maximum(cnt.reindex(v.model_index).fillna(1).values, 1)))
    cp = ROOT / "data/vnc/banc_cable_length.csv"            # skeleton cable length -> area (r = 0.947)
    if cp.exists():
        cab = pd.read_csv(cp).set_index("root_id").cable_um
        cl = cab.reindex(v.bid).values
        est = np.where(np.isfinite(cl) & (cl > 0), np.exp(1.179 + 1.037 * np.log(np.maximum(cl, 1e-3))), est)
    area = np.where(np.isfinite(area) & (area > 0), area, est)
    # normaliser: median area of Pugliese et al.'s measured subnetwork (their a/size, theta*size convention)
    med = float(np.nanmedian(pt.surf_area_um2)) if pp.exists() else float(np.median(area))
    size[v.model_index.values] = area / med
    return size


def main():
    rules = json.loads((ROOT / "data/physiology_rules.json").read_text())
    # FLY_DEFAULT_RATE (calibration experiments): override the rates of the purely approximate rules
    # (catch-all default and 'descending'), leaving every literature/measured rule untouched
    if os.environ.get("FLY_DEFAULT_RATE"):
        dr = float(os.environ["FLY_DEFAULT_RATE"])
        for r in rules["rules"]:
            if r["match"] == {} or r["match"] == {"super_class": ["descending"]}:
                r["rate_hz"] = dr
    if os.environ.get("FLY_VNC_GRADED") == "1":
        # hypothesis test (Pugliese et al. 2025 rate model; graded premotor interneurons in insects): all
        # VNC-intrinsic interneurons non-spiking with graded release
        rules["rules"].insert(0, {"match": {"super_class": ["vnc_intrinsic"]}, "graded": {"r_max_hz": 100.0},
                                  "basis": "hypothesis: VNC interneurons graded (FLY_VNC_GRADED=1)"})
    if os.environ.get("FLY_MN_RATE") == "1":
        # experiment: leg motor neurons as rate units too (as in Pugliese et al.'s model)
        ru = next(r for r in rules["rules"] if "rate_unit" in r)
        rules["rules"].insert(rules["rules"].index(ru) + 1, {"match": {"cell_class": ["leg_motor_neuron"]}, "rate_unit": ru["rate_unit"],
                                  "basis": "experiment: leg MNs as rate units (FLY_MN_RATE=1)"})
    if os.environ.get("FLY_DEFAULT_ADAPT"):
        da = float(os.environ["FLY_DEFAULT_ADAPT"])
        for r in rules["rules"]:
            if r["match"] == {} or r["match"] == {"super_class": ["descending"]}:
                r["adapt_index"] = da
    sigma = rules["noise"]["sigma_mv"]; tau_n = rules["noise"]["tau_ms"]; tau_w = rules["adaptation"]["tau_w_ms"]
    mus, rate, bs, index, d100 = calibrate(sigma, tau_n, tau_w)
    print(f"calibration: rate at mu=0 {np.interp(0, mus, rate):.2f} Hz; step drive for 100 Hz onset {d100:.1f} mV; "
          f"adaptation index at b=0 {index[0]:.2f}, at b=1 {np.interp(1, bs, index):.2f}")
    # neuron table in model order
    comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "super_class", "cell_class", "cell_type"]).drop_duplicates("root_id").set_index("root_id").reindex(comp.index)
    tk = pd.read_csv(ROOT / "data/tastekin2026_flywire_types.tsv", sep="\t", comment="#")
    a.loc[a.index.isin(tk.root_id[tk["class"] == "motor"]), "super_class"] = "motor"
    tab = a.reset_index(drop=True)
    vp = ROOT / "data/vnc/vnc_neurons.csv"
    if vp.exists():
        vn = pd.read_csv(vp)
        vsc = vn.super_class.replace({"ventral_nerve_cord_intrinsic": "vnc_intrinsic", "sensory_ascending": "sensory"})
        vcc = vn.cell_class.replace({"taste_bristle_gustatory_neuron": "gustatory"}).where(
            ~vn.cell_class.isin(["chordotonal_organ_neuron", "campaniform_sensillum_neuron", "hair_plate_neuron",
                                 "bristle_neuron", "taste_bristle_tactile_neuron"]), "mechanosensory")
        tab = pd.concat([tab, pd.DataFrame({"super_class": vsc, "cell_class": vcc, "cell_type": vn.cell_type})], ignore_index=True)
    # DoOR spontaneous rates per glomerulus (spike-recording datasets only)
    sfr = pd.read_csv(ROOT / "data/raw/door/door_sfr.csv")
    ephys = [c for c in sfr.columns if any(k in c for k in ("Hallem", "Bruyne", "Kreher", "Dweck", "Yao", "Goldman", "Kwon", "Montague", "Stensmyr", "Goes", "Ronderos"))]
    sfr["rate"] = sfr[ephys].where(sfr[ephys] > 0).median(1)
    mp = pd.read_csv(ROOT / "data/raw/door/door_mappings.csv", sep=";")
    glo_rate = mp.merge(sfr[["receptor", "rate"]], on="receptor").dropna(subset=["rate"]).groupby("glomerulus").rate.median()
    orn_median = float(glo_rate.median())
    print(f"DoOR: {len(glo_rate)} glomeruli with measured ORN spontaneous rates (median {orn_median:.1f} Hz)")
    out = np.zeros((len(tab), 4), np.float32); out[:, 3] = 1.0
    target_rate = np.full(len(tab), -1.0, np.float32)     # homeostatic set point (Hz); -1 = none
    graded = np.zeros((len(tab), 4), np.float32)            # non-spiking: (r_max Hz, gain Hz/mV, thr mV, 0)
    vnc_size = _vnc_sizes(len(tab)) if any("rate_unit" in r for r in rules["rules"]) else None
    used = {}
    ct = tab.cell_type.fillna("").astype(str).values; cc = tab.cell_class.fillna("").astype(str).values
    sc = tab.super_class.fillna("").astype(str).values
    for i in range(len(tab)):
        for k, r in enumerate(rules["rules"]):
            m = r["match"]
            if "cell_type" in m and ct[i] not in m["cell_type"]:
                continue
            if "cell_class" in m and "super_class" in m:
                if not (cc[i] in m["cell_class"] or sc[i] in m["super_class"]):
                    continue
            elif "cell_class" in m and cc[i] not in m["cell_class"]:
                continue
            elif "super_class" in m and sc[i] not in m["super_class"]:
                continue
            if "rate_unit" in r:
                # Pugliese et al. 2025 VNC rate unit: no noise/offset/adaptation/homeostasis; size-normalised
                # gain and threshold mapped to mV (see _vnc_sizes / data/physiology_rules.json)
                ru = r["rate_unit"]; sz = float(vnc_size[i])
                out[i] = (0.0, 0.0, 0.0, 1.0)
                graded[i] = (ru["fcap_hz"], ru["gain_hz_per_mv"] / sz, ru["threshold_mv"] * sz, 0.0)
                used[k] = used.get(k, 0) + 1
                break
            if "graded" in r:
                # non-spiking neuron: membrane noise, no resting offset/adaptation/homeostasis, graded release
                out[i] = (0.0, sigma, 0.0, tau_w)
                graded[i] = (r["graded"]["r_max_hz"], 0.0, 0.0, 0.0)
                used[k] = used.get(k, 0) + 1
                break
            target = r["rate_hz"]
            if target is None:
                break
            if target == "door_sfr":
                glom = ct[i].replace("ORN_", "")
                target = float(glo_rate.get(glom, orn_median))
            mu = float(np.interp(target, rate, mus)) if target > 0 else -8.0   # far below threshold: silent
            b = float(np.interp(r["adapt_index"], index[::-1], bs[::-1]))
            out[i] = (mu, sigma, b, tau_w)
            target_rate[i] = target if r.get("homeostasis", True) else -1.0
            used[k] = used.get(k, 0) + 1
            break
    out.tofile(ROOT / "data/physiology.bin")
    target_rate.tofile(ROOT / "data/physiology_targets.bin")
    graded.astype(np.float32).tofile(ROOT / "data/graded_release.bin")
    meta = {"n": len(tab), "sigma_mv": sigma, "tau_noise_ms": tau_n, "tau_w_ms": tau_w,
            "rules_used": {rules["rules"][k]["basis"][:60]: v for k, v in sorted(used.items())},
            "calibration": {"mu_mv": mus[::40].round(2).tolist(), "rate_hz": rate[::40].round(2).tolist()}}
    (ROOT / "data/physiology.json").write_text(json.dumps(meta, indent=1))
    for k, v in sorted(used.items()):
        r = rules["rules"][k]
        print(f"  {v:7d} neurons  rate {r.get('rate_hz', 'graded')}  adapt {r.get('adapt_index')}  <- {r['basis'][:70]}")
    print(f"wrote data/physiology.bin ({len(tab)} neurons)")


if __name__ == "__main__":
    main()
