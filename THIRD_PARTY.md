# Third-party data, models and assets

This project builds on the following work. Their licenses and citation requirements apply to the files derived from
them (raw downloads live in `data/raw/`, which is not committed; `tools/fetch_data.py` records every source and
checksum). Please cite the original work when you use this project. Full list with DOI links: [REFERENCES.md](REFERENCES.md).

| Source | Used for | License / terms | Cite |
|---|---|---|---|
| FlyWire connectome v783 and Codex (Princeton), https://codex.flywire.ai | brain wiring, neuron positions, community labels | FlyWire citation guidelines and principles (codex.flywire.ai) | Dorkenwald et al. 2024 Nature; Schlegel et al. 2024 Nature |
| FlyWire annotations, https://github.com/flyconnectome/flywire_annotations | cell types, transmitters, sides | see repository | Schlegel et al. 2024 Nature |
| Shiu et al. brain model, https://github.com/philshiu/Drosophila_brain_model | LIF parameters, reference model, benchmark | MIT | Shiu et al. 2024 Nature |
| BANC brain and nerve cord connectome, https://doi.org/10.7910/DVN/8TFGGB, https://codex.flywire.ai/banc | nerve-cord wiring, annotations, skeleton cable lengths | CC BY 4.0 | Bates et al. 2025 bioRxiv |
| Pugliese et al. VNC CPG model, https://github.com/smpuglie/Pugliese_2026, https://zenodo.org/records/22260924 | VNC rate-model equations, parameters, BANC subnetwork, surface areas | CC BY 4.0 (data) / see repository (code) | Pugliese et al. 2025 bioRxiv |
| Stürner et al. neck connective tables | DN/AN types for joining brain and nerve cord | see publication | Stürner et al. 2025 Nature |
| NeuroMechFly v2 / FlyGym, https://github.com/NeLy-EPFL/flygym, incl. flybody model and musculoskeletal leg model | body meshes and rig (`assets/flygym`, `assets/fly/fly.glb`) | Apache 2.0 (`assets/flygym/LICENSE`) | Wang-Chen et al. 2024 Nat Methods; Lobato-Rios et al. 2022 Nat Methods |
| FlyVis, https://github.com/TuragaLab/flyvis | graded optic lobe model weights | see repository | Lappalainen et al. 2024 Nature |
| Tastekin et al. gustatory connectome | taste neuron types, feeding motor neurons, leg-PER circuit | see publication / bioRxiv | Tastekin et al. 2026 Cell |
| Shiu, Sterne et al. 2022; Sterne et al. 2021 | named feeding neurons | see publications (eLife, CC BY) | Shiu et al. 2022 eLife; Sterne et al. 2021 eLife |
| DoOR 2.0 database, https://github.com/Dahaniel/DoOR.data | olfactory receptor spontaneous rates | see database | Münch & Galizia 2016 Sci Rep |
| Virtual Fly Brain CATMAID (FAFB), https://fafb.catmaid.virtualflybrain.org | skeleton soma positions for cross-checks | see VFB terms | Zheng et al. 2018 Cell |
| Micro-CT eye map, https://github.com/reiserlab/eyemap_T4 | ommatidia positions and directions (`data/ommatidia*.json` are derived) | GPL-3.0 | Zhao et al. 2025 Nature |
| MuJoCo, https://github.com/google-deepmind/mujoco | physics engine (validation tools) | Apache 2.0 | Todorov et al. 2012 |
| Further literature values | physiology rules, see `data/physiology_rules.json` "basis" fields and README | — | as cited there |

Licenses and terms can change; check each source before redistributing derived data.
