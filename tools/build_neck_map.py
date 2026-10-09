"""Neck motor map: real neck motor neurons -> head rotation axes, and the neck proprioceptors.
Source: BANC annotations in data/vnc/vnc_neurons.csv (function / sub_class), FlyWire brain annotations (nerve CV).
  yaw  : ADNM1, ADNM2 (sub_class neck_yaw_motor_neuron, cervical nerve)
  roll : DProN1-4 (neck_roll_motor_neuron), DProN5 (neck_roll_pitch_motor_neuron; roll only, its pitch direction is
         not annotated)
  side: the nerve the axon leaves through (one ADNM soma lies contralateral to its nerve)
  direction: each motor neuron rotates the head toward its own side (yaw: turns toward that side; roll: lowers that
         side). basis: axis from annotation (measured), side convention assumed
  not mapped (data gap, no axis or muscle annotated): DProN6-9 and the 20 brain neck motor neurons of the cervical
         nerve (FlyWire CB0705, CB0706, CB0831, CB0835, CB0838, CB0899, CB0913, CB0916, CB4212)
Sensors: prosternal organ hair plates (SNpp19; head position, Preuss & Hengstenberg 1992) and neck chordotonal organ
neurons; their axis and direction come from tools/neck_sensor_tuning.py.
usage: .venv/Scripts/python tools/build_neck_map.py -> data/neck_motor_map.json"""
import json, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
v = pd.read_csv(ROOT / "data/vnc/vnc_neurons.csv")
ct = v.cell_type.astype(str)
# motor side = the nerve the axon leaves through (the muscle's side), not the soma side
side = v.nerve.astype(str).str.split("_").str[0].where(v.nerve.notna(), v.side.astype(str))
out = {"basis": __doc__.split("Sensors:")[0].strip(), "motor": {}, "sensory": {}, "unmapped": {}}
for axis, types in (("yaw", ["ADNM1", "ADNM2"]), ("roll", ["DProN1", "DProN2", "DProN3a", "DProN3b", "DProN4", "DProN5"])):
    sel = v[ct.isin(types)]
    out["motor"][axis] = {"pos": [int(x) for x in sel[side[sel.index] == "left"].model_index],   # + = toward the left
                          "neg": [int(x) for x in sel[side[sel.index] == "right"].model_index]}
out["sensory"]["prosternal"] = [int(x) for x in v[(v.cell_class == "hair_plate_neuron") & (ct == "SNpp19")].model_index]
out["sensory"]["neck_chordotonal"] = [int(x) for x in v[v.sub_class.astype(str) == "neck_chordotonal_organ_neuron"].model_index]
out["unmapped"]["vnc_neck_mn_no_axis"] = [int(x) for x in v[ct.isin(["DProN6", "DProN7", "DProN8", "DProN9"])].model_index]
a = pd.read_csv(ROOT / "data/raw/annot_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                usecols=["root_id", "super_class", "cell_type", "nerve"])
out["unmapped"]["brain_cervical_mn_root_ids"] = [str(x) for x in a[(a.super_class == "motor") & (a.nerve == "CV")].root_id]
json.dump(out, open(ROOT / "data/neck_motor_map.json", "w"), indent=1)
print({k: {s: len(x) for s, x in m.items()} for k, m in out["motor"].items()},
      {k: len(x) for k, x in out["sensory"].items()}, {k: len(x) for k, x in out["unmapped"].items()})
