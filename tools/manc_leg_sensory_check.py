"""Cross-check of the BANC leg load-sensor (campaniform sensillum) gap against MANC v1.0 (male nerve cord, Janelia;
data/raw/manc/manc-v1.0-neuron-properties.feather, see tools/fetch_data.py).
Prints, per leg segment, MANC's proprioceptor subclasses (FeCO claw/hook/club, trochanteral CS, hair plate, untyped)
next to BANC's annotated classes. Result 2026-10-09: MANC v1.0 also types only the trochanteral CS (2-4 per leg); the
remaining 79-120 proprioceptors per leg segment carry only the generic "proprioceptive" label, so the missing
middle/hind-leg campaniform sensilla cannot be filled from either public table without inventing labels.
usage: .venv/Scripts/python tools/manc_leg_sensory_check.py"""
import pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_feather(ROOT / "data/raw/manc/manc-v1.0-neuron-properties.feather")
s = m[m["class"].astype(str).str.contains("sensory", case=False) & (m.modality.astype(str) == "proprioceptive")]
sub = s.subclass.astype(str)
for seg in ("prothoracic", "mesothoracic", "metathoracic"):
    x = sub[sub.str.startswith(seg + " leg")]
    print(f"MANC {seg} leg proprioceptors:", x.str.replace(seg + " leg", "").str.strip().replace("", "untyped").value_counts().to_dict())
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
b = v[v.super_class == "sensory"]
for leg in ("front", "middle", "hind"):
    y = b[b.sub_class.astype(str).str.startswith(leg + "_leg")]
    print(f"BANC {leg} leg:", y.cell_class.value_counts().to_dict())
