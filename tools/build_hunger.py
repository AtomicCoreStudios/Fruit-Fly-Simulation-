"""Hunger-modulated neurons -> data/hunger_targets.json (model indices) for scripts/fly_taste.gd.
Hunger (fly.hunger, 0 fed .. 1 starved) raises presynaptic release (lif.glsl binding 20) of:
  sugar_grn: all sugar GRNs (labellar, leg-ascending, nerve-cord leg)  - dopamine via DopEcR (Inagaki et al. 2012 Cell)
  central:   G2N-1 (CB0616) and Clavicle (AN_GNG_30)                   - Shiu, Sterne et al. 2022 eLife Fig. 4
Gain = 1 + GAIN_PER_HUNGER * hunger. basis: sites from the literature; size approximate (not measured in spikes)."""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
GAIN_PER_HUNGER = 1.0
comp = pd.read_csv(ROOT / "data/raw/Completeness_783.csv", index_col=0)
a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                usecols=["root_id", "cell_type"]).drop_duplicates("root_id").set_index("root_id").reindex(comp.index)
named = pd.read_csv(ROOT / "data/tables/named_feeding_neurons.csv", comment="#")
central = {}
for r in named[named.hunger_modulated.astype(str).str.startswith("yes")].itertuples():
    central[r.name] = [int(i) for i in np.where(a.cell_type.values == r.flywire_type)[0]]
t = pd.read_csv(ROOT / "data/taste_neurons.csv")
sugar = sorted(int(i) for i in t[t.modality.isin(["sugar", "sugar/low_salt"])]["index"])
out = {"gain_per_hunger": GAIN_PER_HUNGER, "sugar_grn": sugar, "central": central,
       "basis": "sites: Inagaki et al. 2012 (sugar GRNs), Shiu et al. 2022 (G2N-1, Clavicle); gain approximate"}
(ROOT / "data/hunger_targets.json").write_text(json.dumps(out))
print(f"hunger targets: {len(sugar)} sugar GRNs; central " + ", ".join(f"{k} {v}" for k, v in central.items()))
