# Third-party data, models and assets

This project builds on the following work. Their licenses and citation requirements apply to the files derived from
them (raw downloads live in `data/raw/`, which is not committed; `tools/fetch_data.py` records every source and
checksum). Please cite the original work when you use this project, following each provider's rules ("How to cite" in [REFERENCES.md](REFERENCES.md)). Full list with DOI links: [REFERENCES.md](REFERENCES.md).

| Source | Used for | License / terms | Cite |
|---|---|---|---|
| FlyWire connectome v783 and Codex (Princeton), https://codex.flywire.ai | brain wiring, neuron positions, community labels | **CC BY-NC 4.0** (FlyWire public release, https://flywire.ai/guidelines); FlyWire citation guidelines | Dorkenwald et al. 2024 Nature; Schlegel et al. 2024 Nature |
| FlyWire annotations, https://github.com/flyconnectome/flywire_annotations | cell types, transmitters, sides | FlyWire public release, CC BY-NC 4.0 (no license file in the repository) | Schlegel et al. 2024 Nature |
| Shiu et al. brain model, https://github.com/philshiu/Drosophila_brain_model | LIF parameters, reference model, benchmark | MIT | Shiu et al. 2024 Nature |
| BANC brain and nerve cord connectome, https://doi.org/10.7910/DVN/8TFGGB, https://codex.flywire.ai/banc | nerve-cord wiring, annotations, skeleton cable lengths | CC BY 4.0 | Bates et al. 2025 bioRxiv |
| Pugliese et al. VNC CPG model, https://github.com/smpuglie/Pugliese_2026, https://zenodo.org/records/22260924 | VNC rate-model equations, parameters, BANC subnetwork, surface areas | CC BY 4.0 (data) / see repository (code) | Pugliese et al. 2025 bioRxiv |
| Stürner et al. neck connective tables | DN/AN types for joining brain and nerve cord | see publication | Stürner et al. 2025 Nature |
| NeuroMechFly v2 / FlyGym, https://github.com/NeLy-EPFL/flygym, incl. flybody model and musculoskeletal leg model | body meshes and rig (`assets/flygym`, `assets/fly/fly.glb`) | Apache 2.0 (`assets/flygym/LICENSE`) | Wang-Chen et al. 2024 Nat Methods; Lobato-Rios et al. 2022 Nat Methods |
| FlyVis, https://github.com/TuragaLab/flyvis | graded optic lobe model weights | MIT | Lappalainen et al. 2024 Nature |
| Tastekin et al. gustatory connectome | taste neuron types, feeding motor neurons, leg-PER circuit | see publication / bioRxiv | Tastekin et al. 2026 Cell |
| Shiu, Sterne et al. 2022; Sterne et al. 2021 | named feeding neurons | see publications (eLife, CC BY) | Shiu et al. 2022 eLife; Sterne et al. 2021 eLife |
| DoOR 2.0 database, https://github.com/Dahaniel/DoOR.data | olfactory receptor spontaneous rates | see database | Münch & Galizia 2016 Sci Rep |
| Virtual Fly Brain CATMAID (FAFB), https://fafb.catmaid.virtualflybrain.org | skeleton soma positions for cross-checks | see VFB terms | Zheng et al. 2018 Cell |
| Micro-CT eye map, https://github.com/reiserlab/eyemap_T4 | ommatidia positions and directions (`data/ommatidia*.json` are derived) | GPL-3.0 | Zhao et al. 2025 Nature |
| FlyGym 2.1.0 (PyPI) incl. FlyMimic musculoskeletal model, https://github.com/gizemozd/FlyMimic | MuJoCo contact-physics body and musculoskeletal smoke tests (`.venv-flygym`, `tools/flygym_smoke.py`; assets cached in git-ignored `assets/flygym_cache`) | Apache 2.0 (FlyGym); FlyMimic see repository | Wang-Chen et al. 2024; Ozdil et al. 2026 (ICLR, arXiv 2509.06426) |
| MuJoCo, https://github.com/google-deepmind/mujoco | physics engine (validation tools) | Apache 2.0 | Todorov et al. 2012 |
| Further literature values | physiology rules, see `data/physiology_rules.json` "basis" fields and README | — | as cited there |

Licenses and terms can change; check each source before redistributing derived data.
