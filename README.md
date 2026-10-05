# FLY-UI — embodied whole-brain emulation of *Drosophila*

Research aim: a small-scale test of the "Uploaded Intelligence" idea from Pantheon. If you copy a complete nervous system (brain and nerve cord, synapse by synapse from real connectome data) into a simulated body, does the animal's behavior emerge? And where it doesn't, what's missing that the wiring diagram alone can't supply?

The pipeline is scan (connectome) → emulate (spiking brain) → embody (virtual body and world).
The brain and body run in a closed loop in real time.

## Run

1. Both connectomes are already built in `data/`. To rebuild, use the project's own Python environment:
   ```bash
   .venv/Scripts/python tools/build_connectome.py flywire
   ```
   ```bash
   .venv/Scripts/python tools/build_connectome.py synthetic
   ```
2. Open the folder in Godot 4.6 (`F:\GODOT4.6`) and press Play. It needs the
   Forward+ renderer, because the brain runs in a compute shader.
   - The **real FlyWire brain** loads by default.
   - Add `-- --connectome=synthetic` to load the hand-designed stand-in instead.

Controls:
- **L** loom a predator
- **W** wind
- **F** / click: food
- **B** bitter/aversive patch
- **T** move the sun
- **C** camera
- **V** show/hide the brain view
- **1/2/3** time scale
- **R** reset the fly
- **O** optogenetic drive of P9/DNp09 (the forward-walking command neuron)

Test harness (arguments go after `--`):
- `--log` print a report every second
- `--pops` include per-population rates in that report
- `--auto` scripted wind and predator events
- `--threat_at=N` frame at which the predator launches
- `--start=x,z` and `--yaw=a` fly start position and heading
- `--frames=N` quit after N frames
- `--shot=screenshots/x.png` save a screenshot before quitting
- `--connectome=flywire|synthetic` choose the brain
- `--opto` start with P9 optogenetics on
- `--record=recordings/run.csv` write body state plus every population's firing rate every 100 ms of emulated time

Always add Godot's `--quit-after` as a watchdog.

## What is in it

| Layer | File | Notes |
|---|---|---|
| Connectome builder | `tools/build_connectome.py` | Synthetic stand-in, or the real FlyWire v783 connectome |
| Brain emulator | `shaders/lif.glsl`, `scripts/fly_brain.gd` | Leaky integrate-and-fire neurons with Shiu et al. 2024 parameters. Integration step 0.5 ms; synapses stored in sparse row format with fixed-point atomic adds. About 15 ms of GPU time per frame for 139,255 neurons on an RTX 2080. |
| Body, senses, motor | `scripts/fly.gd` | See the sensory and motor lists below |
| World, HUD, brain view | `scripts/main.gd`, `shaders/neuron_view.gdshader` | Every neuron is drawn as a point at its position, coloured by neuropil and brightened by recent spiking |

Senses in `scripts/fly.gd`:
- **Eyes:** 2 × 6 sectors, each with a contrast-adapted tonic channel and a darkening-transient channel.
- **Antennae:** olfactory receptor neurons with adaptation.
- **Johnston's organ:** wind.
- **Taste:** sweet and bitter taste receptor neurons.
- **Sun compass.**

Motor outputs (descending neurons → behaviour):
- P9 → walk forward
- DNa02 → turn
- MDN → walk backward
- Giant fibre → escape takeoff
- MN9 → proboscis extension and feeding

Behaviours verified in test runs:
- Feeding on sugar, which lowers hunger.
- Giant-fibre escape from a looming object, with no false escapes while walking or turning (thanks to the efference copy).
- Wind-driven turning.
- Spontaneous meandering.
- Local search in food odour.
- A heading "bump" in the central complex that tracks the sun.

Odour-guided approach works only partly. The fly stays in the plume and circles near the food, but it does not reliably land on it.

## Downloaded data (`data/raw/`)

| File | Source | What it is |
|---|---|---|
| `Completeness_783.csv`, `Connectivity_783.parquet` | github.com/philshiu/Drosophila_brain_model | FlyWire v783: 138,639 proofread neurons, 15,091,983 connections, 54.5M synapses, signed by predicted transmitter |
| `model.py`, `utils.py`, `example.ipynb`, `figures.ipynb`, `sez_neurons.pickle`, `environment*.yml` | same | Shiu et al. 2024 reference model. MN9 = root 720575940660219265. The pickle holds data only and was checked before loading. |
| `annot_Supplemental_file1..5` | github.com/flyconnectome/flywire_annotations | Schlegel et al. 2024: super class, class, subclass, cell type, side, soma and anchor positions, neurotransmitter; plus hemilineages and hemibrain matches |

## Real-brain mode: what maps to what

| Input or output | Real FlyWire neurons |
|---|---|
| Eyes, tonic channel | R7/R8 photoreceptors plus lamina neuron L3 |
| Eyes, darkening transient | Lamina neuron L2 |
| Eyes, brightening transient | Lamina neuron L1 |
| Food odour | Receptor neurons of glomeruli DM1, DM2, DM4, DP1m, VA2, VM2 |
| Aversive odour | Receptor neurons of glomeruli V (CO2) and DA2 (geosmin) |
| Wind | Johnston's organ wind/gravity neurons |
| Taste | Sugar and bitter taste neurons, by side |
| Motor outputs | DNa02 (turning), DNp09 (forward), MDN (backward), DNp01 (giant fibre), MN9 = CB0701 (proboscis) |

- The eyes are split into sectors using real neuron positions: lamina upright, medulla inverted by the optic chiasm. **This retinotopic mapping is approximate.**
- Input groups are also ready but not yet driven by anything in the world: water taste, temperature (hot/cold), humidity (moist/dry), hearing, ocelli, and the dorsal rim area.
- **Model fidelity:** matches Shiu et al. 2024 — leaky integrate-and-fire neurons, 0.275 mV per synapse, 1.8 ms synaptic delay, synaptic input reset on spike, Poisson sensory drive of 68.75 mV. One difference: the time step is 0.5 ms instead of their 0.1 ms, so it can run in real time.
- **Body calibration:** how descending-neuron rates map to body movement is in the `motor` block of `data/connectome_flywire.json`. Edit it freely.

## First findings with the real brain

- **No spontaneous walking.** Nothing in the wiring activates DNp09 without an internal-state drive. With P9 optogenetics the fly walks.
- **Left turn bias.** DNa02_L fires at about 60–120 Hz while DNa02_R stays silent, even with no odour present. The fly circles left.
- **No looming escape.** LPLC2 and the giant fibre stay silent to the looming predator. The simplified lamina drive probably doesn't produce the direction-selective motion signals (T4/T5 cells) that loom detection needs.
- **APL runs at 300+ Hz.** In reality APL is a non-spiking, graded neuron, which a leaky integrate-and-fire model can't represent.
- **EPG (compass) neurons are silent.** Nothing drives the ring-neuron and sun-compass pathway yet.

## Important honesty notes

- **The synthetic connectome** (`--connectome=synthetic`) It uses FlyWire's neuron count and the real neuropils and pathways, but the synapse-level wiring was written by hand to produce these behaviours. It is **not** an upload of a real fly.
- **The real brain** is the actual FlyWire wiring. It still runs a simplified neuron model and uses a hand-built body.
- **What no connectome contains.** A UI in the Pantheon sense would need all of the following, and none of them is in a wiring diagram:
  - learning and synaptic plasticity
  - neuromodulators (dopamine, octopamine)
  - gap junctions and synaptic delays
  - dendritic nonlinearities
  - the ventral nerve cord (the fly's "spinal cord": FlyWire covers the head brain only)
  - the brain's current state (what makes it a particular individual)

## Anatomical body and compound eyes (Blender, in progress)

Built by scripts that run inside Blender (`tools/blender/`) and plain Python (`tools/`). Open `blender/fly_body.blend`.

| Part | Source | Basis |
|---|---|---|
| Body meshes (69 segments) | NeuroMechFly v2 / FlyGym (Apache-2.0), micro-CT of one adult female, `assets/flygym/` | measured (left side); right side mirrored = approximate |
| Body frames and joint axes | FlyGym `rigging.yaml` → `data/flygym_rig.json` (`tools/flygym_rig_to_json.py`) | measured |
| Standing pose | FlyGym `pose/neutral/pitch_roll_yaw.yaml` | literature |
| Ommatidia (1,709: 852 right, 857 left) | Zhao et al. 2025 *Nature* micro-CT eye (`data/raw/eyemap_T4/`, reiserlab/eyemap_T4) | optical axes measured |
| Facet → FlyWire column | Zhao's lens↔Mi1 map of the FAFB eye + FlyWire Codex column assignment (Matsliah et al. 2024) | right eye measured; left eye approximate (mirrored lattice) |
| R1–6 → cartridge | FlyWire v783 connectivity: each R1–6 goes to the column of its strongest L1–L3 targets (5,749 of 7,172 placed) | measured |
| Yellow / pale | none in FlyWire; random 70/30 | approximate |
| Dorsal rim area | Zhao's DRA Mi1 columns | right measured, left mirrored |
| Facet positions on the body | micro-CT lenses fitted onto the NeuroMechFly eye mesh (scale 0.956, rotation 3.8°, median residual 4 µm) | approximate |

Outputs:
- `data/ommatidia.json`: per facet: side, hex (p,q), column_id, optical axis, DRA, subtype, `basis`, and the FlyWire root IDs of its column (R7, R8, L1–L5, C2, C3, T1–T3, Mi1/4/9, Tm1/2/3/4/9/20/21, T4a–d, T5a–d, R1–6).
- `data/ommatidia_placement.json`: facet position and axis in the head frame (mm).

Checks:
- The Codex (p,q) ↔ Zhao (i,j) link was verified two independent ways (lattice overlap 770/779; photoreceptor-less edge columns 40/56).
- The median interommatidial angle comes out at 4.8° (right) and 4.7° (left); the literature value is about 4.5–5°.
- A typical column holds one neuron of each of the 31 columnar types plus exactly 6 R1–6 (neural superposition).

Caveats:
- The eye micro-CT, the FlyWire brain and the NeuroMechFly body come from three different flies.
- Only the right optic lobe's column map is tied to measured eye data.
- Rebuild with `.venv/Scripts/python tools/build_eye_map.py`, then run `tools/blender/build_body.py` and `tools/blender/place_ommatidia.py` in Blender.

## Compound-eye vision in Godot (facet vision, default with the FlyWire brain)

The fly in Godot is the NeuroMechFly micro-CT body (`assets/fly/fly.glb`, exported from `blender/fly_body.blend`). Every segment is a named node (`c_head`, `lf_tibia`, `l_arista`, …) where further sense organs can be attached. Leg and wing motion is cosmetic; there is no nerve-cord motor model yet.

Pipeline:
1. **Cube map.** Six 90° cameras (48 px per face) form a cube map. It is locked to the `c_head` node, centred between the eyes. The fly's own body is on render layer 2 and hidden from these cameras.
2. **Facet sampling.** `shaders/eye.glsl` samples each of the 1,709 measured ommatidia with a 5° Gaussian acceptance cone. Photoreceptor adaptation is Weber-like, with τ = 1 s.
3. **Photoreceptor drive.** The shader writes a Poisson rate into the per-neuron input buffer (new binding 14 in `lif.glsl`) for every real FlyWire R1–6, R7 and R8 of that facet's column (8,062 neurons). The kernel and receptor lists come from `data/eye_drive.json` (`tools/build_eye_drive.py`).
4. **Transmitter sign.** Photoreceptor output synapses are set **inhibitory (histamine)** in `tools/build_connectome.py`. FlyWire's transmitter predictor has no histamine class. basis: literature.
5. **Lamina resting bias.** Lamina L1–L5 get a resting bias of `--lamina_bias` (default 9 mV). They are graded cells in reality. basis: approximate.
6. **Columnar resting bias.** Other columnar optic-lobe types (Mi, Tm, C, T1–T5) can get `--columnar_bias=mV` (default 0). basis: approximate, free parameter.

Other approximations:
- RGB render with blue standing in for UV.
- One viewpoint for both eyes.
- Graded photoreceptors are represented as spike rates.
- The fly body casts no shadow (its 0.3 mm self-shadow flickered in the shadow map).

New arguments:
- `--vision=facets|sectors` (sectors is the old 2 × 6 model)
- `--lamina_bias`, `--columnar_bias`
- `--tethered`: body held in place; giant-fibre spikes are counted as escapes
- `--threat_angle=deg`: 0 = head-on, 90 = from the left
- `--eye_dump=file.csv`: per-facet luminance; plot it with `tools/plot_eye_view.py`
- `--probe=data/loom_probe.json`: per-column traces (`tools/make_loom_probe.py`)
- `--cam_dist`: camera distance

### Loom test (2026-09-30)

Run with `tools/run_loom_test.py --bias 0 5 6 6.5`. It records `recordings/loom_*.csv` / `*_summary.csv`, and per-column probes to `recordings/probe_b*.csv`.

- **Stimulus:** fly tethered; a dark sphere (radius 0.7 cm) launched from 9 cm to the fly's left at 7 cm/s.
- **Result:** no escape in any condition. LPLC2, LC4 and DNp01 (giant fibre) stay at 0 Hz.

Where the signal goes, in the 12 columns that view the predator (the control columns ahead stay flat):

| Stage | bias 0 mV | bias 6.5 mV | Real-fly expectation |
|---|---|---|---|
| R1–6 | 35 → 9 Hz | same | photoreceptors respond to darkening ✔ |
| L1 / L2 | 5 → 22 / 3 → 16 Hz | same | depolarize to darkening ✔ |
| Tm1 / Tm2 / Tm9 (OFF) | 0 → 3 / 1 / 0 | 2 → 20 / 20 / 7 | OFF cells excited ✔ (only with bias) |
| Mi1 / Tm3 (ON) | 0 | 8 → 3 / 2 → 1 | suppressed by a dark object ✔ (only with bias) |
| T5 (OFF motion) | 0 | 0 → up to 7 (T5d), 4 (T5c) | responds to a dark edge ✔ weak |
| T4 (ON motion) | 0 | ≈0 | little response ✔ |
| LPLC2 → DNp01 | 0 | 0 | should fire → escape ✘ |

Interpretation:
- The measured eye and the real wiring give the correct ON/OFF polarity at every stage up to T5.
- The signal is lost in two places:
  1. **At the first medulla synapse.** Lamina input delivers only about 0.6–1.5 mV of mean depolarization to Tm/Mi cells, against a 7 mV spike threshold. In reality these are graded, non-spiking synapses, which a spiking LIF model with the Shiu et al. parameters cannot represent.
  2. **At T5 → LPLC2.** A few weak, non-direction-selective T5 spikes don't reach LPLC2's threshold. Direction selectivity needs cell-type-specific time constants (delays between Tm9 and Tm1/Tm2), and this model gives every synapse the same 1.8 ms delay and 5 ms time constant.
- **Likely next steps:**
  - Run the optic lobe as a graded (rate) network with cell-type time constants (FlyVis, Lappalainen et al. 2024 is the data-driven source of those parameters), feeding LPLC2/LC4 into the spiking central brain.
  - Or add per-type time constants to the LIF shader.

## Graded optic lobe: FlyVis hybrid (default)

The optic lobe can now run as the **FlyVis** graded network (Lappalainen et al. 2024, *Nature*; github.com/TuragaLab/flyvis, MIT), pretrained model `flow/0000/000`. It runs on the GPU in `shaders/flyvis.glsl`: one instance per eye, each with 65 cell types and 45,669 cells.

What FlyVis supplies:
- **Connectivity** is measured, but from FIB-25/FIB-19 medulla data averaged into per-type filters, not from FlyWire.
- **Resting potentials, time constants and synapse strengths** are model-fitted: they were trained so the network computes optic flow.
- **Export check:** the exported network matches FlyVis's own simulator to 1×10⁻⁶ (`tools/flyvis/export_flyvis.py`). FlyVis is exactly convolutional, so 2,355 shared filter entries reproduce all 1,513,231 synapse groups.

How it connects to the rest of the model:
- **Input:** each FlyVis column reads its facet's luminance. The lattice is laid onto each measured eye (`tools/flyvis/map_eye_lattice.py`) and oriented so the model's T4/T5 preferred directions match biology (Maisak et al. 2013).
  - Remaining direction errors are 2–10° for most subtypes; T5d is 25–30° and T4c is 40–44°, which reflects FlyVis's own tuning.
  - 712 of 721 columns (right eye) and 674 (left) sit on a real facet.
- **Output:** every FlyVis cell whose type is in FlyWire's column map drives the real FlyWire neuron(s) of that type in that column, as a Poisson rate: `gain × (relu(V) − relu(V_grey))`, with `--flyvis_gain`, default 60 Hz/unit, approximate. That covers 28 types and **37,627 FlyWire neurons** (L1–L5, C2, C3, T1–T3, Mi1/4/9, Tm1/2/3/4/9/20, T4a–d, T5a–d).
- **Downstream:** everything beyond those cells (LPLC2, LC4, the giant fibre and the rest of the central brain) remains the spiking FlyWire LIF model.
- **Options:** `--optic_lobe=lif` switches back to the spiking-only optic lobe.

Rebuild:
1. `.venv/Scripts/python tools/flyvis/export_flyvis.py`
2. `.venv/Scripts/python tools/flyvis/map_eye_lattice.py`
3. `.venv/Scripts/python tools/flyvis/export_godot.py`

`tools/flyvis/_env.py` patches a Windows file-locking bug in FlyVis's storage library (datamate) at runtime.

Cost: about 20% more GPU time per simulated step. It runs in real time at about 27 fps.

### Loom test with the FlyVis optic lobe

Fly tethered; predator from the left. Records are in `recordings/loom_flyvis_g*`; the figure is `screenshots/flyvis_loom_g120.png`.

| | gain 60 | gain 120 | gain 240 |
|---|---|---|---|
| Columns facing the predator: T4a/b/c/d, T5a/d peak | — | 62 / 53 / 34 / 16, 28 / 22 Hz (from 0–1) | higher |
| Control columns | flat | flat | flat |
| LPLC2 | 0 | 0 | 0 |
| LC4 | not loom-specific | brief peak 18 Hz at the very end, tonic about 10 Hz | tonic about 23 Hz |
| Giant fibre (DNp01) | none | 1 false alarm (also in the no-predator control) | false alarms in the control |

- **Result:** with graded dynamics, the real eye geometry produces a strong, loom-specific, all-direction (expanding) T4/T5 response in exactly the columns that see the predator. Under the purely spiking optic lobe this signal was absent.
- **Where it stops:** the transmission from **T4/T5 → LPLC2** in the spiking FlyWire model.
  - Each LPLC2 has about 890 input synapses, with only about 30–40 per T4/T5 subtype, spread across a wide receptive field.
  - Its largest input, **Tm5f (67 synapses per cell), isn't in FlyVis** and stays silent.
  - It receives inhibition from PVLP011 (−46 per cell).
  - The estimated loom-driven depolarization is about 4 mV, against a 7 mV threshold.
- **Raising the gain doesn't help:** it only adds tonic LC4 drive and false giant-fibre spikes.
- **What would plausibly fix it:**
  - Graded or dendritic integration in LPLC2 itself; real LPLC2 dendrites combine layer-specific outward motion non-linearly.
  - Including Tm5f/TmY inputs.
  - Per-type LIF parameters.
  - None of these are in a wiring diagram.

## Graded optic lobe beyond FlyVis, and two-compartment visual projection neurons (default)

FlyVis now covers 47 cell types (46,229 FlyWire neurons). Twenty types that FlyWire's column map lacks were placed by connectivity, using `tools/flyvis/assign_columns.py`. That method recovers 83% of known columns exactly and 96% to within one column.

Every other FlyWire optic-lobe neuron is a **graded unit**: about 31,600 of them (LPi, Tm5f, Y, TmY, Li, Dm, Pm, Lawf, lobula-plate tangential cells, …). So is the **dendrite of every visual projection neuron**: 8,038 of them (LPLC2, LC4, LPLC1, LC6, LC12, …). Code is in `tools/graded/` and `shaders/graded.glsl`.

How the graded units are defined:
- **Dynamics:** the same equations as FlyVis.
- **Connectivity:** FlyWire synapse counts, with signs from the predicted transmitter. basis: measured.
- **Synapse strength:** FlyVis's fitted per-synapse strength for the presynaptic type, adjusted three ways:
  - scaled by FlyVis's own rule that cells with many inputs have weaker synapses (strength ∝ total inputs^−0.38);
  - converted from FIB-25 to FlyWire synapse counts (κ = 0.87);
  - multiplied by one global factor (α = 0.42) so the recurrent graded block has the same largest real eigenvalue (0.89) as FlyVis's fitted network, which keeps it stable.
  - basis: model-derived.
- **Resting level:** set so each unit sits at threshold under uniform grey light. basis: approximate.

How the graded part connects to the spiking brain:
- **Projection neurons are two-compartment.** The graded dendrite drives the spiking FlyWire soma as a bias current of c × V. The default c = 19 mV per unit is the mean-field equivalent of the spiking model's own synapses (`--vpn_c`). Their axonal outputs (to the giant fibre and so on) are ordinary spiking FlyWire synapses.
- **No double counting:** the spiking model's optic-lobe-internal synapses are silenced (7.7 million edges, in `data/lif_w_graded.bin`).
- **Switches:** `--graded=0` turns this off; `--optic_lobe=lif` gives the spiking-only optic lobe.

Rebuild everything with `bash tools/rebuild_vision.sh flow/0000/000`.

### Validation against physiology (offline, `tools/graded/validate.py`)

**Choosing the FlyVis model.** All 50 pretrained FlyVis models were screened on established T4/T5 physiology, without reference to any loom result (`tools/flyvis/screen_ensemble.py`, `pd_check.py`). The two criteria:
- strong direction selectivity with a weak response to stationary flashes;
- T4 and T5 of the same subtype preferring the same direction (Maisak et al. 2013).

How the candidates fared:
- Several high-scoring models (021, 022, 036) **violate the T4/T5 co-tuning constraint**; for example, 021's T5c and T5d are reversed.
- Models 000 and 005 pass both criteria.
- On the stimulus suite (below), **000 matches more of Klapoetke's LPLC2 tests**, so it is the default.

**Wiring check, independent of FlyVis** (`tools/graded/radial_test.py`). FlyWire T4/T5 → LPLC2 wiring is driven with an idealized radial-motion pattern. **All 102 right LPLC2 cells prefer outward motion, by a median factor of 25**, which reproduces Klapoetke et al.'s anatomical result in our eye geometry.

**Response tests** (model 000; peak population response relative to the loom):

| Stimulus | LPLC2 | Expected (Klapoetke 2017) |
|---|---|---|
| dark loom | 1.00 | strong |
| receding disc | 0.40 | ≈0 ✔ (reduced) |
| inward cross | 0.23 | ≈0 ✔ |
| wide-field translation | 0.36 | ≈0 ✔ (reduced) |
| outward cross | 0.39 | strong ✘ (weak) |
| luminance-matched dimming | 1.47 | ≈0 ✘ |

- **LC4** is silent to the dark loom and responds to brightening. In this FlyVis model, LC4's main input (T2) is a pure ON cell; real LC4 responds to dark looms (Ache et al. 2019).
- **Where the remaining failures come from:** FlyVis T4/T5 output. In model 000, T5 responds to stationary darkening almost as much as to motion. Across the whole eye, the T5 subtypes are unevenly tuned (in model 005, T5a dominates every OFF edge).
- **Bottom line:** the connectome wiring is radially correct. The motion inputs from any single FlyVis model aren't clean enough to reproduce the full selectivity.

### In-world loom test

Fly tethered, predator from the left (`tools/run_loom_test.py --optic graded --vpn_c 19 38 76 --control`):

| Coupling c | LPLC2 (approached side) | LC4 | Giant fibre |
|---|---|---|---|
| 19 mV/unit | peak 0.8 Hz, loom only (control 0) | tonic 5–8 Hz | silent |
| 38 mV/unit | peak 3.8 Hz, loom only | tonic 23–34 Hz | only a start-up transient, also in the control |
| 76 mV/unit | 8–13 Hz, loom only | tonic 50–75 Hz | fires every ~2 s, also in the control (false alarms) |

- **LPLC2** is loom-specific for the first time and scales with c.
- **No loom-triggered escape** occurs without false alarms. LC4 is tonically driven by the bright arena (an ON-cell artifact inherited from FlyVis T2), and it reaches the giant fibre before LPLC2 does.
- **Most likely improvements:**
  - an ensemble average of several co-tuned FlyVis models, as Lappalainen et al. recommend for predictions;
  - the newer whole-optic-lobe connectome-constrained models trained on FlyWire or the male CNS (not done here).
