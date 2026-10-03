"""Diagnostic: which neurons carry sugar -> MN9 in the exact Shiu et al. reference (mirror on), written as a
Godot probe file so the embodied brain's rates for the same neurons can be compared.
usage: diag_sugar_path.py prep  -> recordings/sugar_path_ref.json + recordings/sugar_path_probe.json
       diag_sugar_path.py cmp <probe csv>"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reference_lif import LIF, ROOT, MN9, SUGAR_R

tn = pd.read_csv(ROOT / "data/taste_neurons.csv")
stim_ids = SUGAR_R   # Shiu et al. Fig. 1 set (Godot --taste_set=shiu)
if sys.argv[1] == "prep":
    m = LIF("shiu", mirror=False)
    idx = m.index.reindex(stim_ids).dropna().astype(int).values
    c = sum(m.run([(idx, 100.0)], 1000.0, seed=s) for s in range(3)) / 3.0
    a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "cell_type", "side", "top_nt"]).drop_duplicates("root_id").set_index("root_id")
    act = np.where(c >= 15)[0]
    act = act[~np.isin(act, idx)]
    rows = {f"{a.cell_type.get(m.ids[i], '?')}_{str(a.side.get(m.ids[i], '?'))[:1]}_{a.top_nt.get(m.ids[i], '?')[:4]}_{i}": [float(c[i]), int(i)] for i in act}
    rows = dict(sorted(rows.items(), key=lambda kv: -kv[1][0]))
    (ROOT / "recordings/sugar_path_ref.json").write_text(json.dumps(rows, indent=0))
    probe = {k: [v[1]] for k, v in rows.items()}
    probe["GRN_stim"] = [int(i) for i in idx]
    (ROOT / "recordings/sugar_path_probe.json").write_text(json.dumps(probe))
    print(f"{len(rows)} non-stimulated neurons >= 15 Hz in the reference at 100 Hz (Shiu sugar set)")
else:
    ref = json.loads((ROOT / "recordings/sugar_path_ref.json").read_text())
    d = pd.read_csv(sys.argv[2]); d = d[d.t_ms > 1000]
    print(f"GRN_stim {d['GRN_stim'].mean():.0f} Hz")
    out = [(k, v[0], d[k].mean()) for k, v in ref.items()]
    for k, r, g in out:
        print(f"{k:45s} ref {r:6.1f}  godot {g:6.1f}")
