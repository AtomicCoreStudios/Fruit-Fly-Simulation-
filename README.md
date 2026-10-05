# FLY-UI — embodied whole-brain emulation of *Drosophila*

Research aim: a small-scale test of the "Uploaded Intelligence" idea from Pantheon. If you copy a complete nervous system (brain and nerve cord, synapse by synapse from real connectome data) into a simulated body, does the animal's behavior emerge? And where it doesn't, what's missing that the wiring diagram alone can't supply?

The pipeline is scan (connectome) → emulate (spiking brain) → embody (virtual body and world).
The brain and body run in a closed loop in real time.

## Current status (2026-10-05)

| Part | State | Basis |
|---|---|---|
| Brain | FlyWire v783 (139,255 neurons) as Shiu et al. LIF, with conductance synapses, spontaneous activity, adaptation, homeostasis, hunger neuromodulation, and graded APL / patchy AL LNs. Sugar → MN9 and other Shiu benchmarks reproduced. | measured wiring; literature and assumed physiology (see sections below) |
| Nerve cord | BANC VNC (22,681 resident neurons) joined to the brain: 161,320 neurons and ~15.7M edges in total. VNC interneurons and leg motor neurons are rate units (Pugliese et al. 2025 model). | measured wiring; literature model |
| Eyes and vision | 1,709-facet compound eyes → FlyVis + graded optic lobe → VPNs | measured / literature |
| Taste, smell, wind | per-neuron sense organs on the NeuroMechFly body | measured mapping; approximate response curves |
| Legs | 391 real leg motor neurons → muscles → joints (torque model); 886 proprioceptors feed back | measured mapping; assumed mechanics (calibration ledger) |
| Walking | Walking CPG reproduced exactly offline. Live: DNg100 drives a 5–8 Hz leg rhythm; swing/stance alternation and tripod are partial; free-walking speed 0.08–0.21 mm/s (real 10–30). **Not walking yet.** | see "Neuromuscular legs" |
| Escape | giant fibre → TTMn (gap junction) works | literature |
| Leg sugar → PER | known gap (see Phase 5b) | — |

Everything below is in chronological order. Dated sections describe the state at that date; this table is the current summary.

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
- `--physiology=0` turn off spontaneous activity and adaptation (pure Shiu et al.)
- `--phys_ablate=noise,mu,adapt,homeo` turn off parts of the intrinsic physiology
- `--homeostasis` learn homeostatic offsets during the run and save them at the end; use it with `--clean_air`
- `--clean_air` no odour, wind or taste stimuli
- `--opto` start with P9 optogenetics on
- `--record=recordings/run.csv` write body state plus every population's firing rate every 100 ms of emulated time
- `--probe=groups.json --probe_out=out.csv` per-frame activity of chosen neurons (plus leg joint angles, stance and foot height when the legs are on)
- `--act_tau=5` time constant (ms) of the activity readout (default 60; use 5 at `--fixed-fps 120` for rhythms)
- `--synapses=current` Shiu et al.'s current-like synapses instead of the default conductance synapses
- `--drive=i,j:Hz;k:Hz` Poisson drive on chosen model neurons (e.g. descending neurons)
- `--silence=i,j` and `--release_gain=i,j:g` set chosen neurons' transmitter release (diagnostics)
- `--sugar_release=x` force the sugar-GRN release gain (otherwise set by hunger)
- `--legs=0` cosmetic legs instead of the neuromuscular model; `--legs_log`, `--legs_lift_test` leg diagnostics
- `--locomotion=legs` the body moves only by its stance feet (default: descending-neuron rate kinematics)

Always add Godot's `--quit-after` as a watchdog.

## What is in it

| Layer | File | Notes |
|---|---|---|
| Connectome builder | `tools/build_connectome.py` | Synthetic stand-in, or the real FlyWire v783 connectome |
| Brain emulator | `shaders/lif.glsl`, `scripts/fly_brain.gd` | Leaky integrate-and-fire neurons with Shiu et al. 2024 parameters (plus conductance synapses, intrinsic physiology, graded and rate units: see later sections). Integration step 0.5 ms; synapses stored in sparse row format with fixed-point atomic adds. About 17 ms of GPU time per frame for 161,320 neurons on an RTX 2080 (50 fps). |
| Legs | `scripts/fly_legs.gd`, `tools/build_leg_map.py` | Neuromuscular legs and proprioceptors (Phase 6) |
| Taste | `scripts/fly_taste.gd` | Per-neuron taste organs (Phase 3) |
| Body, senses, motor | `scripts/fly.gd` | See the sensory and motor lists below |
| World, HUD, brain view | `scripts/main.gd`, `shaders/neuron_view.gdshader` | Every neuron is drawn as a point at its position, coloured by neuropil and brightened by recent spiking |

Senses in `scripts/fly.gd`:
- **Eyes:** 2 × 6 sectors, each with a contrast-adapted tonic channel and a darkening-transient channel.
- **Antennae:** olfactory receptor neurons with adaptation.
- **Johnston's organ:** wind.
- **Taste:** sweet and bitter taste receptor neurons.
- **Sun compass.**

Motor outputs (descending neurons → behaviour; walking is still driven this way by default, the neuromuscular legs are opt-in for locomotion via `--locomotion=legs`):
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

## First findings with the real brain (2026-09-30; historical, several since addressed)

- **No spontaneous walking.** Nothing in the wiring activates DNp09 without an internal-state drive. With P9 optogenetics the fly walks.
- **Left turn bias.** DNa02_L fires at about 60–120 Hz while DNa02_R stays silent, even with no odour present. The fly circles left.
- **No looming escape.** LPLC2 and the giant fibre stay silent to the looming predator. The simplified lamina drive probably doesn't produce the direction-selective motion signals (T4/T5 cells) that loom detection needs. (Addressed in part later by the FlyVis and graded optic lobe; see those sections.)
- **APL runs at 300+ Hz.** In reality APL is a non-spiking, graded neuron. (Fixed 2026-10-04: APL is now a graded neuron.)
- **EPG (compass) neurons are silent.** Nothing drives the ring-neuron and sun-compass pathway yet.

## Important honesty notes

- **The synthetic connectome** (`--connectome=synthetic`) It uses FlyWire's neuron count and the real neuropils and pathways, but the synapse-level wiring was written by hand to produce these behaviours. It is **not** an upload of a real fly.
- **The real brain** is the actual FlyWire wiring. It still runs a simplified neuron model and uses a hand-built body.
- **What no connectome contains.** A UI in the Pantheon sense would need all of the following, and none of them is in a wiring diagram:
  - learning and synaptic plasticity (still missing)
  - neuromodulators (dopamine, octopamine): only hunger → sugar-pathway release is modelled so far
  - gap junctions and synaptic delays: delays modelled; gap junctions only for the documented giant-fibre ones
  - dendritic nonlinearities (still missing; graded and conductance neurons are a partial step)
  - the ventral nerve cord (the fly's "spinal cord"): now added from BANC, a different fly than FlyWire's brain
  - the brain's current state (what makes it a particular individual): still missing

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

## Core brain fidelity: matched to Shiu et al. exactly (2026-10-03)

`tools/reference_lif.py` is an exact re-implementation of the Shiu et al. 2024 Brian2 model (`data/raw/model.py`): 0.1 ms step, exact linear integration, Poisson input into the membrane voltage, no refractory period for stimulated neurons. It revealed that our old GPU scheme under-drove the brain badly. Poisson input went into the synaptic conductance and was Euler-integrated, so at 100 Hz sugar input MN9 reached 30 Hz instead of 75.

`shaders/lif.glsl` now uses Shiu's semantics: exact integration, Poisson kicks to the voltage, stimulated neurons non-refractory, synaptic input reset on spike. The only remaining difference is the 0.5 ms step (needed for real time), which also rounds the 1.8 ms delay to 2.0 ms.

Benchmark (Shiu et al. Fig. 1): Shiu's 20 "sugar" GRNs (v783 IDs) stimulated at f Hz, MN9 rate R/L in Hz.

| f | exact reference (0.1 ms) | Godot GPU (0.5 ms) |
|---|---|---|
| 40 | 2.3 / 2.0 | 3 / 4 |
| 60 | 29 / 23 | 23 / 25 |
| 80 | 61 / 47 | 56 / 42 |
| 100 | 75 / 53 | 87 / 56 |
| 150 | 100 / 65 | 107 / 58 |

The two agree to within about 15%, which is about the size of the trial-to-trial spread.

**Note:** loom and vision results recorded before this date used the old scheme. Re-run them before comparing.

## Taste organs and the feeding motor system (Phase 3)

Data and tools:
- **Taste neurons:** `tools/build_taste.py` → `data/taste_neurons.csv`, `data/taste_organs.json`. Every FlyWire gustatory neuron (407) is assigned an organ, side, modality and sensor.
- **Sensor placement:** `tools/blender/place_taste.py` puts the sensors on the body: 62 labellar taste bristles (L/I/S rows), 73 taste pegs, 30 leg-tarsus sensors and 1 pharyngeal sensor.
- **Physics:** in `scripts/fly_taste.gd`, a sensor reads chemistry only when it physically touches a patch. Each neuron fires by its modality's dose–response, from literature ranges in `data/grn_response.json` (basis: approximate).
- **GPU input:** rates go straight into the brain through `shaders/sensor.glsl`.

Where each piece comes from:

| Item | Source | Basis |
|---|---|---|
| Labellar bristle types and modality | Tastekin et al. 2026 types (flywire_annotations v3.2.0) | measured |
| Leg neurons LgAG1–9 | LgAG2 = sugar (Gr61a match), LgAG1 = bitter (Gr33a); other types aversive / unknown cluster; leg of origin (fore vs mid/hind) from the male CNS (Tastekin et al. 2026) | literature |
| Pharyngeal neurons PhG1–16 | PhG1 = sugar-like (Gr64e); PhG3/4 = water; PhG2 and the aversive cluster = putative aversive (Tastekin et al. 2026, Suppl. Table 2 → `data/tastekin2026_flywire_types.tsv`) | literature |
| Taste pegs | no ligand known → not driven | — |
| Neuron → individual bristle, positions on the labellum | dealt by co-expression rules | approximate |
| Feeding motor neurons, 24 types (MN1–13, CEM, MNx) | Tastekin et al. 2026 Suppl. Table 2; muscle roles from McKellar et al. 2020 | measured / literature |
| Proboscis kinematics | MN9 vs MN1 → rostrum; MN4a (+MN6) vs MN1 → haustellum; MN8 → labellum spreading (exposes pegs); MN11/12 → cibarial pump. Activation = rate/(rate+30 Hz), 60 ms time constant | approximate |
| Labellum–substrate contact | rostrum ≥ 0.7 and haustellum ≥ 0.5 extended; pegs also need spreading ≥ 0.3. NeuroMechFly's neutral pose stands taller than a feeding fly, and posture isn't simulated | approximate |
| Ingestion | labellum on sucrose and the pump active | — |

### Findings

1. **Sugar alone evokes a full, coordinated feeding motor program from the wiring.** With Shiu's sugar set at 100 Hz:
   - MN9 (rostrum protraction) 50–80 Hz
   - MN4a (haustellum extension) 26 Hz
   - MN6 (labellar extension) 19–34 Hz
   - MN8 (labellar spreading) 21–76 Hz
   - MN11D (cibarial pump) 165–175 Hz
   - **MN1 (the retractor) silent**

   Nothing about this coordination was programmed.
2. **Shiu et al.'s 21 "sugar" neurons are mixed, and all on one side.** Re-typed with v3.2.0, they are 9 sugar, 2 sugar/low-salt, 5 high-salt/heavy-metal and 4 putative-attractive neurons, one is missing from v783, and every one is annotated `side = left`. Stimulated alone (`tools/decompose_shiu_sugar.py`), the 9 sugar neurons give MN9 34 Hz at 100 Hz input. The high-salt and putative-attractive subsets also drive MN9 at 200 Hz.
3. **Left and right labellar taste neurons are not equivalent in FlyWire v783.** Right-annotated labellar neurons have about **half the reconstructed output synapses** (median 127 vs 240, in every type), and their full sugar set drives MN9 to 0 Hz even at 100 Hz. The full left sugar set drives MN9 at 64 Hz at 60 Hz input (`tools/sugar_sides.py`). This points to incomplete reconstruction of one labellar nerve, not real biology.
4. **No proboscis extension from leg sugar, on the sucrose patch or in the bench test.**
   - On a 200 mM sucrose patch, all 12 leg sugar neurons (LgAG2) fire at about 32 Hz. MN9 stays at 0.
   - Even at 200 Hz, LgAG2 gives MN9 only about 2 Hz.
   - In real flies tarsal sugar reliably triggers extension, mostly via leg taste neurons that stay in the **ventral nerve cord**, which FlyWire doesn't contain. Only the ascending subset reaches the brain.
   - So in this model the fly cannot start feeding from its legs. That's a missing-data result: the nerve cord (MANC/BANC) is needed to close the loop.

Bench: `tools/run_taste_bench.py --set shiu|sugar|leg_sugar|bitter`.

## Mirrored right labellar taste neurons (on by default)

**The problem.** In FlyWire v783, right-annotated labellar taste neurons have about half the output synapses of the left ones:

| Class | Neurons L/R | Median output synapses L/R |
|---|---|---|
| Labellar bristles | 108 / 105 | 240 / 127 |
| Taste pegs | 37 / 36 | 149 / 74 |

Their cable is only about 14% shorter, but their synapse density is 43% lower (364 vs 634 per mm). Labial mechanosensory neurons in the same nerve, and the pharyngeal and leg taste neurons, are symmetric. Engert et al. 2022 report EM-volume misalignments that prevented labellar taste-neuron reconstruction in FAFB's left hemisphere. That is the side FlyWire labels right, because FAFB was imaged as a mirror image.

**The fix** (`tools/mirror_grn.py` → `data/grn_mirror_patch.parquet`; turn it off with `FLY_NO_MIRROR=1`):
- Each right taste neuron is topped up to the left-side mean for its type.
- Each type keeps its own pattern of same-side vs opposite-side partners, copied from the left. Bitter neurons reach the midline ring; sugar, water and salt neurons stay on their own side (Engert et al. 2022).
- Synapses are added only to the mirror-image partner neurons (same type, opposite side), in proportion to the synapses those partners already receive.
- Measured synapses are never removed.
- Taste-to-taste contacts are added once, from the presynaptic side.

Result: about 14,500 output and 2,000 input synapses added; the right-side median goes from 101 to 221 synapses (left: 204). basis: approximate.

**Functional check** (reference model, `tools/sugar_sides.py --mirror`). MN9 R/L in Hz:

| Sugar set | Neurons | 60 Hz input | 100 Hz input |
|---|---|---|---|
| left | 33 | 64 / 40 | 96 / 51 |
| right, before | 24 | 0 / 0 | 0 / 0 |
| right, after | 24 | 21 / 26 | 44 / 61 |

The opposite-side preference flips as a mirror image should. The right side stays weaker because FlyWire types fewer of its neurons as sugar.

## Spontaneous activity, adaptation and homeostasis (Phase 5, 2026-10-03, on by default)

Shiu et al.'s model is silent without input and has no adaptation. Real fly neurons fire spontaneously and adapt. Added per neuron (`shaders/lif.glsl`, `tools/build_physiology.py`, rules in `data/physiology_rules.json`):
- **Membrane noise:** Ornstein–Uhlenbeck, σ = 2 mV, τ = 5 ms. basis: approximate (Gouwens & Wilson 2009).
- **Resting offset μ:** calibrated so the isolated neuron fires at its cell class's measured spontaneous rate. Values and basis:

  | Cell class | Rate | Basis |
  |---|---|---|
  | ORNs | per receptor, from DoOR spike recordings (median 10 Hz) | measured (Münch & Galizia 2016; Hallem & Carlson 2006) |
  | PNs | 4 Hz | Wilson et al. 2004 |
  | Kenyon cells | 0.2 Hz | Turner et al. 2008 |
  | MBONs | 6 Hz | Hige et al. 2015 |
  | Giant fibre, escape/flight motor neurons, APL | silent | |
  | Everything else | about 1 Hz | approximate |

- **Spike-frequency adaptation:** AdEx-style current, τ_w = 300 ms. Its step is calibrated to a per-class adaptation index; Kenyon cells adapt strongly (0.3), motor neurons weakly (0.85). basis: approximate (Nagel & Wilson 2011).
- **Homeostatic intrinsic plasticity:**
  - A 60 s warm-up (`--homeostasis --clean_air`) learns a per-neuron offset that brings each neuron in the network to its target rate. It is saved to `data/homeostasis_offsets.bin` and loaded automatically.
  - Offsets: 5th–95th percentile −3.4 to +0.5 mV.
- **Synapse signs by Dale's principle:** each neuron's transmitter is `known_nt`, else FlyWire's neuron-level `top_nt` (Eckstein et al. 2024).
  - ACh is excitatory; GABA, glutamate and histamine are inhibitory.
  - Dopamine, serotonin, octopamine and tyramine have no fast effect.
  - Shiu et al.'s per-connection signs counted monoamines as excitatory. `FLY_SIGNS=shiu` restores them.

Two wiring corrections were needed to make this hold up:
1. **No spike-driving inputs onto sensory afferents.**
   - ORNs, GRNs and mechanoreceptors start their spikes in the periphery. FlyWire/BANC synapses onto their axon terminals (ORN–ORN, LN → ORN, GABA → GRN) act presynaptically on release (Olsen & Wilson 2008; Root et al. 2008), so 185,359 such edges were removed (`FLY_AFFERENT_INPUTS=1` keeps them).
   - Presynaptic gain control itself is not modelled.
2. **Input-resistance correction for giant integrators only.**
   - Shiu's 0.275 mV per synapse is kept for neurons with up to 6,000 input synapses, the range of the circuits Shiu et al. validated (MN9 has 5.6–5.9k).
   - Above that, a passive-membrane correction w × (6000 / N_inputs) is applied. It affects about 0.5% of neurons, mostly multiglomerular antennal-lobe local neurons (9–17k inputs) and APL.
   - Without it those cells saturate on spontaneous ORN input (local neurons 61 Hz, APL 70–96 Hz, even with homeostasis).
   - The FlyVis power law over all neurons (γ = 0.38) also tames them but weakens sugar → MN9 about 5× by scaling down the premotor hubs (CB0553, CB0493, DNge062).
   - No single power law fits both benchmarks (it would need γ > 1), hence the threshold. basis: biophysics + approximate threshold; `FLY_SIZE_NREF`, `FLY_SIZE_GAMMA`.

Also changed:
- The old 5 Hz (ORN) and 10 Hz (Johnston's organ) Poisson baselines are dropped when physiology is loaded, since they would double-count spontaneous activity.
- `--clean_air` turns off odour, wind and taste stimuli for resting calibration, like in-vivo recordings in clean air.

Results:
- **Resting rates** (clean air, no vision, 10 s; `tools/run_rest_test.sh`): measured against target.

  | Population | Measured | Target |
  |---|---|---|
  | ORNs | 12.4 | 10 |
  | PNs | 4.5 | 4 |
  | AL local neurons | 4.7 | 2 |
  | Kenyon cells | 0.1 | 0.2 |
  | MBONs | 4.7 | 6 |
  | DANs | 1.9 | 2 |
  | LH local neurons | 1.1 | 3 |
  | LH centrifugal | 1.9 | 3 |
  | APL | 0 | 0 |
  | MN9 | 0.2 | 0.5 |

  - Rates are averages over cell types.
  - Per neuron, AL local neurons have median 0 Hz and 13% above 10 Hz. The fastest are the serotonergic lLN2T_b (~100 Hz), which have no fast output here.
- **Global runaway:** gone. Bilateral 200 Hz sugar leaves APL at 0, AL local neurons at about 2 Hz and Kenyon cells at 0.1 Hz.
- **Sugar → MN9:** bilateral labellar sugar at 30/60/100/200 Hz gives MN9_R 8/14/20/38 Hz and MN9_L 5/7/11/20 Hz, rising steadily with input.
  - Shiu's exact model gives about 29/75/116 Hz at 60/100/200 Hz.
  - Much of that comes from a recurrent pharyngeal premotor loop (CB4055 / CB0910 / CB0707 / CB0708) that ignites to about 100 Hz in Shiu's model. With bilateral input it ignites 8,002 neurons at ≥ 15 Hz. With adaptation that loop stays moderate, so our MN9 is lower. Which is closer to the real fly is not settled by data; MN9 has not been recorded in Hz.
- **Giant fibre:** opto at 100 Hz drives TTMn at 44–57 Hz through the gap junction. Escape still works.
- **Leg sugar** (embodied, standing on the food patch): MN9 rises from 0.2 to 2.8 Hz but there is no visible PER. Real flies extend the proboscis to tarsal sugar, so the leg → PER route is still too weak (open issue).
- **Ablations** (`--phys_ablate=noise,mu,adapt,homeo`, γ = 0.38 build, 100 Hz bilateral sugar, MN9_R):

  | Configuration | MN9_R |
  |---|---|
  | All physiology on | 2 Hz |
  | No homeostasis | 18 Hz |
  | No adaptation | 18 Hz |
  | Neither | 36 Hz |
  | No noise | 0 Hz |
  | Physiology off | 0 Hz |

  Noise lets weak synaptic drive reach threshold.

Diagnostics: `tools/diag_sugar_path.py` compares the reference sugar pathway neuron by neuron with the embodied brain.

Note on running Godot: on this PC windowed Godot started freezing on vsync (2026-10-03, even with an empty project). All test scripts now launch it with `--disable-vsync --fixed-fps 30`, which gives about 33 ms of simulation per frame, as before.

## Leg sugar → proboscis extension (Phase 5b, in progress, 2026-10-03)

The circuit, from the literature:
- Tastekin et al. 2026, Fig. 9B: the appetitive leg taste type **LgLG4** (nerve cord, Gr64f) → ascending neuron **AN01B004** (FlyWire AN_GNG_162) → **Bract** descending neurons and Sink & Synch → **Roundup** → **MN9**.
- The direct leg-to-brain type **LgAG2** joins the same targets.

New tools and data:
- **Named feeding neurons in v783** (`data/tables/named_feeding_neurons.csv`):
  - Source: FlyWire Codex community labels.
  - Cross-checked by soma position against the CATMAID skeletons (`tools/data_named_feeding.py`, which agreed on every neuron both methods resolved).
  - Names → types: G2N-1 = CB0616, Clavicle = AN_GNG_30 (an ascending neuron), Fdg = CB0038, Rattle = CB0499, Roundup = CB0553, Rounddown = DNge080, Bract1/2 = DNge174/DNge173, Sink & Synch = CB0434, Foxglove = CB0890, Bluebell = DNg60, and others.
- **Better brain/nerve-cord joining** (`tools/build_vnc.py`):
  - Cell types come as left/right homolog sets. The FlyWire→BANC type correspondence learned from confident matches is used to pair the remaining members within each type.
  - +127 ascending and +53 sensory ascending neurons are now joined (1,266/1,736 and 181/581), e.g. the left Dandelion (AN_GNG_68 = AN13B002).
- **Presynaptic release gain** (`lif.glsl` binding 20, `FlyBrain.set_release_gain`): neuromodulation of transmitter release.
  - Hunger now raises release at the sites the literature identifies (`tools/build_hunger.py` → `data/hunger_targets.json`), with gain = 1 + hunger (×1.8 when hungry):
    - all 370 sugar GRNs: dopamine via DopEcR (Inagaki et al. 2012)
    - G2N-1 and Clavicle: the only hunger-modulated central nodes found by Shiu, Sterne et al. 2022 (Fig. 4)
  - basis: sites from the literature, gain approximate. Benchmarks keep gain 1; `--sugar_release=X` forces it.
  - `--silence=i,j,...` sets chosen neurons' release to 0, for ablation diagnostics.

Findings:

| Test | Result |
|---|---|
| All 293 nerve-cord leg sugar GRNs at 100 Hz → AN01B004 | 4–44 Hz |
| → Bract2 (DNge173), Roundup (CB0553) | 1–2 Hz |
| → MN9 | ≤ 2 Hz |
| Shiu et al.'s exact model, AN01B004 driven directly | needs ~200 Hz for MN9 18 Hz |
| Leg sugar also activates | glutamatergic DNxl094 (583 synapses back onto AN01B004) and GABAergic DNg103: negative feedback |
| Sugar-GRN release ×2 / ×4, labellar (30 Hz bilateral) | MN9 5 → 13 → 20 Hz |
| Sugar-GRN release ×2 / ×4, leg | ≤ 4 Hz |
| Embodied, standing on 200 mM sucrose, hungry (GRNs + G2N-1 + Clavicle ×1.8) | MN9 1–2 Hz, proboscis peaks at 0.28 only briefly |
| Labellar sugar 30 Hz bilateral, fed → hungry | MN9 5 → 18 Hz (strong hunger effect, as in Shiu et al. 2022) |
| Leg 100 Hz with DNxl094, DNg103 and IN09A001 silenced | MN9 ≤ 1.4 Hz (feedback is not the block) |

- AN01B004 provides only 2–8% of Bract1/2's inputs (241 and 146 synapses).
- The leg route needs about 7 hops to reach the feeding motor neurons, 2 more than the labellum (Tastekin et al.).
- Sensory hunger gain explains the labellar effect of starvation but not tarsal PER. The missing gain is central.
- Both of Shiu et al.'s hunger nodes are on the labellar route: Clavicle's inputs are labellar LB3 GRNs, not leg GRNs.
- Conclusion so far: with Shiu-calibrated synapse weights, the connectome's leg arc is too weak in the feed-forward direction to trigger PER on its own, even with the literature hunger mechanisms. This is either a model limitation (point neurons, uniform weight per synapse, glutamate always inhibitory) or a missing hunger mechanism on the leg relay.

### Testing the model assumptions on the leg route (option 1, 2026-10-03)

**Glutamate sign** (`FLY_GLU_SIGN`, diagnostic). Rest values are without homeostasis:

| Variant | Leg 100 Hz → MN9 | Rest MN9 | Labellar 100 Hz → MN9 |
|---|---|---|---|
| −1 (inhibitory, current) | 9 | 6.5 | 51 |
| 0 (no fast effect) | 21 | 9.8 | 51 |
| +1 (excitatory) | 7 | 2.4 | 8 |
| Shiu per-connection signs | 6 | 3.6 | 29 |

- With its own homeostatic warm-up, glutamate-neutral is worse: labellar 30 Hz → MN9 0.9 Hz, leg ≈ 0.
- Homeostasis lowers the whole network's excitability to compensate for the missing inhibition.
- Conclusion: the glutamate sign does not explain the leg gap.

**Strength at the leg relay** (`--release_gain` on the 8 AN01B004 / AN_GNG_162, hungry):

| AN01B004 release | Leg 100 Hz → MN9 (R) |
|---|---|
| ×1 | 3.7 |
| ×2 | 3.4 |
| ×4 | 7.1 |
| ×8 | 29 |

- Leg PER would need roughly 8× stronger synapses at this one relay, which is implausible as the only cause.

**The actual block is the resting state of the feeding motor chain** (`tools/diag_rest_drive.py`).
- Homeostatic offsets:

  | Neuron | Offset |
  |---|---|
  | MN9 | −26 to −28 mV |
  | AN01B004 | down to −30 mV |
  | Roundup | −10 to −12 mV |
  | Rounddown | −6 to −7 mV |
  | Bract2 | about −6 mV |
  | Network median | 0 mV |

- Without homeostasis, MN9 fires about 6 Hz at rest, although its net input is inhibitory:
  - inhibitory CB0862 and CB0903 at 7–11 Hz
  - excitatory DNge062 and Roundup at about 4 Hz
- With current-based synapses, this balanced bombardment produces large voltage fluctuations, so MN9 fires from noise. Homeostasis cancels that with an unphysiologically large hyperpolarization, which buries weak inputs. The strong labellar route still gets through; the 7-hop leg route does not.
- Real intrinsic plasticity shifts excitability by a few mV, and real synapses are conductance-based: inhibition shunts and excitation saturates near its reversal potential, which shrinks such fluctuations.

## Conductance-based synapses (default since 2026-10-03)

The model:
- **Driving forces:** each synapse now acts through a conductance with a driving force (`shaders/lif.glsl` `cond`).
  - E_exc = 0 mV (nicotinic ACh; measured)
  - E_inh = −75 mV (Cl⁻ channels: GABA-A/Rdl, GluCl, histamine-gated; approximate)
- **Calibration:** weights keep Shiu et al.'s 0.275 mV per synapse, normalised at mid-subthreshold, v_n = (v_rest + v_th)/2 = −48.5 mV, where his weight was fitted to behaviour. So inside the operating range a synapse has Shiu's effect. Excitation saturates toward 0 mV; inhibition shunts and reverses at −75 mV.
- **Integration:** exact exponential relaxation with the total conductance, using midpoint conductances over each 0.5 ms step.
- **Switches:** `--synapses=current` restores Shiu's current-like synapses. Each synapse model has its own homeostasis file (`data/homeostasis_offsets.bin`; `..._current.bin`).

Validation in the pure Shiu setting (no intrinsic physiology), 20 right sugar GRNs → MN9 R/L:

| Input | Exact reference, current | Reference, conductance (`tools/reference_lif.py` `cond`) | GPU, conductance |
|---|---|---|---|
| 100 Hz | 75 / 53 | 69.5 / 52 | 69.3 / 50.5 |
| 200 Hz | 116 / 72 | 121 / 82 | 119 / 78 |

- Normalising at rest (−52 mV) instead gives only 23 / 16 at 100 Hz.

With physiology and a fresh homeostatic warm-up:

| Check | Result |
|---|---|
| Resting rates | ORN 12.4, PN 4.8, AL local neurons 5.3, Kenyon cells 0.1, MBON 5.2, APL 0 |
| Bilateral sugar 30 / 100 / 200 Hz → MN9_R | 3.4 / 23 / 32 Hz |
| Same, hungry, at 30 Hz | 19 Hz |
| Giant fibre 100 Hz → TTMn | 51 Hz |

- AN01B004's offsets shrank from up to −30 mV to between +0.1 and −6.6 mV.
- MN9 still needs −28 to −30 mV, Roundup −9 to −11, Bract2 −6.
- Leg sugar still does not reach MN9: ≤ 1.6 Hz, and in the embodied hungry fly 1.1 Hz.
- So the large resting drive onto the feeding motor chain is not only a current-synapse artefact. It comes from the network's spontaneous activity converging on MN9: about 1 Hz assumed for every uncharacterised central neuron, with thousands of inputs.

### Where the leg-route gain is lost (2026-10-03)

**Driving each stage directly** (`--drive=indices:Hz`, no other input), MN9 R response:

| Driven stage | Reference (Shiu, no physiology) 30 / 100 Hz | This model 30 / 100 Hz |
|---|---|---|
| Roundup (CB0553) | 68 / 148 | 21 / 88 |
| Bract (DNge173/174) | 19 / 58 | 0.5 / 6.7 |
| AN01B004 (AN_GNG_162) | 5 / 52 | 4 / 20 |
| G2N-1 (CB0616) | 0 / 54 | 1 / 9 |

- Bract and G2N-1 are each sufficient for PER in real flies (Shiu, Sterne et al. 2022).
- In this model the interneuron stages lose 3–8× gain compared with Shiu's model. The cause is the intrinsic-physiology layer, chiefly the large negative homeostatic offsets on the feeding chain (MN9 about −29 mV, Roundup about −10, Bract2 about −6).
- These neurons are tonically driven at rest by the spontaneous network, and homeostasis holds them at their low set points.

**Variants tested** (each with its own 60 s warm-up):

| Variant | Result |
|---|---|
| Default rate 1 → 0.3 → 0.1 Hz (uncharacterised neurons) | MN9 offset −29 → −18 → −12 mV; leg route unchanged (≤ 0.5 Hz) |
| Homeostasis capped at ±5 mV | AL runs away (local neurons 75 Hz, PNs 26 Hz); Bract → MN9 2× stronger; leg route still ≈ 3 Hz |
| No default adaptation | Homeostasis re-balances; Bract → MN9 weaker (3.9 Hz); leg route ≈ 0 |

- The main build keeps a 1 Hz default rate, adaptation index 0.6, and the −40/+15 mV homeostasis range.

Summary: the leg route works in Shiu's raw model when its interneurons are driven, but leg GRNs cannot drive them hard enough (AN01B004 reaches 0–44 Hz). In the model with physiology, every stage is also damped. The real fly is quiet at rest but highly excitable along this chain. A noisy point-LIF homeostatically pinned to fixed set points cannot be both.

## Non-spiking (graded) neurons (2026-10-04, on by default)

- **Which neurons:** APL (2; Papadopoulou et al. 2011; Amin et al. 2020) and the patchy antennal-lobe local neurons lLN2P_a/b/c (34; Schenk & Gaudry 2023 eNeuro; R32F10 patchy LNs, Sizemore et al. 2022). Both are non-spiking in recordings.
- **Dynamics:**
  - These neurons never spike, reset or go refractory; their membrane integrates continuously, with membrane noise.
  - They release transmitter stochastically at f(v) = r_max / (1 + exp(−(v + 45)/2)), with r_max = 100 Hz. The release curve is approximate.
  - Implementation: `shaders/lif.glsl` binding 23, `data/graded_release.bin`, rule `graded` in `data/physiology_rules.json`.
- **Excluded:** graded neurons have no resting-rate target, adaptation or homeostasis.

Results after a fresh warm-up:
- **lLN2P:** release about 57 events/s at rest (near the top of the curve; they integrate ORN/PN input without spike resets).
- **APL:** tonic graded inhibition at about 5 events/s (previously forced silent). Kenyon cells 0.2 Hz; PNs 3.4 Hz (was 4.8).
- **Unchanged:** sugar → MN9 (bilateral 100 Hz → 23/11 Hz) and giant fibre → TTMn (52 Hz).
- **Not fixed:**
  - The spiking AL hubs il3LN6 and lLN2F_b still sit at the −40 mV homeostasis clamp.
  - They are spiking types (Seki et al. 2010 type I), so making lLN2P graded could not tame them.
  - The leg route is unchanged.

## Neuromuscular legs (Phase 6, in progress, 2026-10-05)

**Motor side** (`tools/build_leg_map.py` → `data/leg_motor_map.json`):
- All 391 BANC leg motor neurons are assigned, from their muscle-target annotations, to six joint degrees of freedom per leg: coxa protraction and adduction, trochanter, femur rotation, tibia, tarsus.
- `scripts/fly_legs.gd`:
  - Firing rate → muscle activation: a = r/(r + 30 Hz), filtered with 30 ms.
  - The agonist and antagonist pools set a joint equilibrium angle, and the NeuroMechFly skeleton joints relax to it with a 30 ms time constant.
  - Ranges and time constants are approximate.
- These replace the cosmetic leg swing (`--legs=0` restores it).

**Sensory side:** 886 proprioceptors drive their real BANC neurons every frame:
- femoral chordotonal organ claw (tibia position, flexion- and extension-tuned)
- hook (movement direction)
- club (movement or vibration)
- hair plates (coxa and trochanter angle near the joint limits)
- campaniform sensilla (muscle-force strain proxy)

Tuning is approximate (Tuthill & Wilson 2016; Mamiya et al. 2018).

**Walking rhythm (CPG): reproduced offline, partly live**

Target: Pugliese et al. 2025 bioRxiv. Driving DNg100 produces a leg motor rhythm via E1 = IN17A001 → E2 = INXXX466 → I1/I2 (IN16B036 / IN19A007) → E1.

1. **Exact reproduction of their model** (`tools/pugliese_repro.py`):
   - Inputs: their BANC front-leg subnetwork, weights and surface areas (`data/raw/pugliese`: Zenodo 22260924 and github smpuglie/Pugliese_2026 @10e7661, CC-BY-4.0).
   - Result: leg motor neuron rhythm 0.5–0.66 at 12.5–17.5 Hz.
   - With our weights and signs swapped in on their neurons: 0.65 at 13.8 Hz. Our synapse data are fine.
2. **Their rate model on our whole BANC nerve cord** (`tools/vnc_rate_model.py`, 25,227 neurons, stimulating the right-soma DNg100 135733, whose axon targets the left legs):

   | Size normalisation | Result |
   |---|---|
   | Measured surface area | E1/E2/I2 and motor neurons rhythmic, 0.96–0.98 at 14 Hz |
   | (synapses/median)^0.60 | rhythmic, 1.0 at 16 Hz |
   | None | runaway, no rhythm |

   - Size normalisation (larger neurons less excitable) is required.
3. **Ported to the live GPU brain:**
   - The 12,827 nerve-cord interneurons are now rate units, parameters mapped from the paper:
     - gain 21.82 Hz/mV ÷ size; threshold 0.344 mV × size above rest; r_max 200 Hz
     - size = surface area where measured, else synapses^0.60 (fit r = 0.90)
     - no noise, offset, adaptation or homeostasis
   - Transmission is smooth: quanta of weight/20 (`RATE_Q`).
   - Motor neurons, ANs, DNs and sensory neurons stay spiking.
   - Result: the CPG interneurons E1/E2/E3 now oscillate together, but weakly (0.25–0.37 at 3–5 Hz). Frequency does not yet rise with drive, and the leg motor neurons do not yet follow.
   - Remaining differences from the offline model: spiking motor neurons with homeostasis, the 5 ms synaptic filter and 1.8 ms delay, conductance synapses, and tonic input from brain, ascending and sensory neurons.
   - Validation with rate units:

     | Check | Result |
     |---|---|
     | Resting rates | unchanged |
     | Bilateral sugar 100 Hz → MN9 | 20 / 12 Hz |
     | Giant fibre → TTMn | 52 Hz |
     | Speed | 50 fps |

### CPG tuning steps (2026-10-05)

Offline, adding the live model's synaptic kinetics to their rate model (`RM_TAU_SYN=5`, `RM_DELAY=1.8`) slows the rhythm from 14 to about 4 Hz but keeps it (0.83). The slow live rhythm therefore comes from synaptic timing.

Live tests:
- **Conductance synapses on the rate units halve the rhythm.** So rate units now take linear, current-based input (excitation minus inhibition) as in the published model, while the spiking brain keeps conductance synapses (`shaders/lif.glsl`). With this, the live CPG is strong: E1/E2/I1/I2 at 0.57–0.61, 6.3 Hz.
- **Spiking leg motor neurons do not follow the rhythm** (≤ 0.25). Leg motor neurons are now firing-rate units too, as in Pugliese et al.; this is an abstraction, since real MNs spike. TTMn keeps its spiking escape rule.
- **The legs do not step yet.** Joint excursions under DNg100 drive are 0.003–0.09 rad, at 2–3 Hz, with no tripod phase pattern (`tools/analyze_gait.py`).
- **Per motor neuron, left front leg, DNg100 300 Hz** (`recordings/mn_lf.csv`):
  - Only 7 of 69 motor neurons are active, consistent with Pugliese et al. ("DNg100 typically recruited 2–10 leg MNs").
  - The active ones are strongly modulated (s.d. ≈ mean).
  - They are all one in-phase stance-like synergy (coxa retraction, trochanter extension, tibia flexion); the antagonists stay silent.
  - The muscle model averages activation over the whole pool, so 2 of 6 active neurons barely move the joint.
  - The gap has therefore moved to the biomechanics. Small insect legs moving in air need little force, and passive joint torques dominate (Hooper et al. 2009 J Neurosci). The "fraction recruited → fraction of range" muscle model is too crude.
- **Torque-based leg mechanics (`fly_legs.gd`):**
  - Motor-unit forces now add up.
  - Each fully active motor neuron gives 0.4 rad of excursion against passive stiffness (`DTHETA_MN`, approximate calibration).
  - Joint limits are soft (tanh), and the joint is overdamped with passive return (Hooper et al. 2009).
- **Result, tethered** (`screenshots/gait_dng100.png`):
  - With DNg100 driven, the legs move continuously and rhythmically: 4–20 Hz joint power 3–4× rest, peaking at about 5 Hz on several legs.
  - Swing in the 3.5–9 Hz band is 0.12–0.21 rad peak-to-peak (rest 0.01–0.06); total excursions reach 0.75 rad.
  - Real stepping is about 1–1.5 rad at 7–15 Hz with a tripod pattern. Not yet reached: amplitude is about a sixth of real, the rhythm is slower and irregular, and there is no inter-leg tripod coordination.
- **Next:** antagonist (swing-phase) recruitment, inter-leg coupling, and proprioceptive feedback while stepping.

### Swing/stance, coordination and free walking (2026-10-05)

1. **Free walking** (`--locomotion=legs`, `FlyLegs.odometry`):
   - Model: stance feet (foot height ≤ rest + 5% of reach, body frame) grip the ground without slipping, and the body moves by minus their mean motion (translation and yaw). It is a kinematic contact model without physics forces.
   - Result: with DNg100 driven, the legs move the body (4.9 mm path in 6.8 s, 5× rest) but the net displacement stays ≈ 0 (≤ 0.06 mm/s; real flies walk 10–30 mm/s). Feet push back and forth without a clean swing phase.
2. **Swing/stance recruitment:**
   - Offline, their model with our synaptic kinetics recruits 20 front-left motor neurons in two clusters about 180° apart: swing (promotors, trochanter and tibia flexors, tarsus depressors) and stance (remotors, trochanter extensors).
   - The stance cluster appears only if unmeasured neurons are given the median size. With the data-based size estimate (area ∝ synapses^0.60, r = 0.90; normalised by the median of the paper's measured subnetwork) only the swing cluster remains, at 7.1 Hz offline and 8 Hz live. Live and offline therefore agree.
   - Live, the right-soma DNg100 (which drives the left legs) gives swing-cluster rhythm at 8 Hz, in the real 7–15 Hz range. Any substantial left-soma DNg100 drive switches the left front leg to a slow, stance-dominated pattern.
   - In insects, stance is reinforced by load feedback from campaniform sensilla (Zill et al. 2004). BANC wiring from leg campaniform sensilla to the motor pools is mixed: direct excitation of retractors and trochanter motor neurons, but net inhibitory 2-hop pathways.
   - Ground load now drives the campaniform neurons in stance (`LOAD_HZ`, approximate). It does not yet produce propulsive stepping.
3. **Inter-leg coordination** (tethered, bilateral DNg100 150 Hz; CTr phase in the 4–9 Hz band):

   | Leg pair | Phase difference | Consistency |
   |---|---|---|
   | Left front vs left middle | −160° | 0.52 |
   | Left middle vs left hind | 178° | 0.51 |
   | Left front vs right front | −153° | 0.5 |
   | Left front vs right middle | none consistent | 0.03 |

   - Ipsilateral neighbours and the front legs alternate, as in a tripod gait. The cross-body tripod partner is not yet locked in phase.

### Real neuron sizes, descending-neuron screen, combinations (2026-10-05)

**Neuron sizes** (`tools/build_banc_cable.py` → `data/vnc/banc_cable_length.csv`):
- Cable length of all 114,999 BANC skeletons, computed from the SWC files of Bates et al. 2025 (Harvard Dataverse doi:10.7910/DVN/8TFGGB, `neuron_skeletons.zip`, md5 f7d16e8e…, CC-BY-4.0; user approved the 215.6 MB download). Codex's BANC attribute table has empty morphology columns.
- Surface area predicted from cable length: log area = 1.037 log cable + 1.179, r = 0.947 on 4,286 neurons with measured area.
- Coverage: 19,368 of 22,681 VNC residents.
- Size priority for rate units: measured area > cable-predicted > synapse-count estimate, normalised by the median of Pugliese et al.'s measured subnetwork.
- With these sizes, right DNg100 alone recruits a swing burst (promotors, trochanter flexors) and one remotor in anti-phase (−178°) at 7.1 Hz.

**Descending-neuron screen** (offline, each neuron alone; left front leg):

| Neuron(s) | Recruits | Rhythm |
|---|---|---|
| DNb08 left-soma (98130, 131703) | swing | 12 Hz, 0.99–1.0 |
| DNb08 right-soma 47118 | stance | 11 Hz, 0.21 |
| DNa02 left 92992 | stance | 10 Hz, 0.46 (24 Hz mean) |
| Right DNg100 | both | 7 Hz, 0.8 |
| DNp09, MDN, DNb02, DNp42 | little or nothing | — |

**Combinations** (offline):
- Right DNg100 + left DNa02: coxa remotors (−90° to −129°) and promotors plus trochanter flexors (≈ +155°) alternate at 7.1 Hz. This is the first swing/stance alternation in the motor output.

**Live free walking** (bilateral DNg100 ± DNa02 ± DNb08, ground-load feedback on):
- Net 0.08–0.21 mm/s, against 10–30 mm/s in real flies. Propulsive stepping is not yet achieved.

**Foot friction / physics contact:**
- The current contact model is no-slip (effectively infinite friction). Realistic friction and adhesion need a force-based physics body; NeuroMechFly does this in MuJoCo.
- Deferred until the motor pattern produces clean swing/stance, since friction cannot create propulsion the legs do not generate.

## Calibration ledger (motor side): what is data, what is set by hand

Kept so that any walking result can state exactly how much came from the connectome and how much from hand-set values. Basis: **measured** = from data; **literature** = published model or parameter; **assumed** = an approximate value I chose; **tuned** = adjusted while looking at walking. Nothing on the motor side has been tuned to walking yet.

| Item | Value | Basis | Walking depends on it? |
|---|---|---|---|
| Wiring: BANC VNC synapses, signs (Dale), DN/AN joining | — | measured | yes |
| Motor neuron → muscle → joint DoF | BANC annotations | measured | yes |
| VNC interneuron dynamics | Pugliese et al. rate model (τ 20 ms, gain, θ, f_cap 200 Hz, W × 0.03) | literature | yes |
| Neuron size for rate units | measured area / cable-predicted (r = 0.947) / synapse fit | measured + fit | yes (stance recruitment depends on it) |
| Leg motor neurons as rate units (not spiking) | abstraction, as Pugliese et al. | literature model | yes (spiking MNs did not follow the CPG) |
| Rate units take linear (current) input; brain uses conductances | as the published model | literature model | yes (conductance halved the rhythm) |
| Synaptic filter 5 ms, delay 1.8 ms | Shiu et al. values | literature | yes (sets the 4–8 Hz frequency) |
| Muscle activation: R_HALF 30 Hz, τ 30 ms | — | assumed | yes |
| DTHETA_MN 0.4 rad per motor neuron | — | assumed | yes (amplitude) |
| Joint range and τ_joint 30 ms | NeuroMechFly-like ranges | assumed | yes |
| Stance rule: foot ≤ rest + 5% of reach | no-slip contact | assumed | yes |
| LOAD_HZ 40 Hz campaniform ground load | — | assumed | under test |
| Proprioceptor tuning curves (claw, hook, club, hair plate) | — | assumed | under test |
| Descending command (which DNs, how strong) | search over walking DNs (DNg100, DNb08, DNa02, DNg97, DNp09) | will be **tuned** (step A) | yes |

**Free-walking diagnosis (2026-10-05)** (`stance_<leg>` and `foot_h_<leg>` columns in probe CSVs):
- All six feet are never down together (0% of the time).
- Instead, legs sit in tonic postures. Left middle is held up 98–100% of the time; left front and right middle are planted 98–100%.
- Foot-lift test (`--legs_lift_test`): trochanter flexion lifts the foot (+0.23 to +0.28 of reach per 0.3 rad), so the swing group can lift. Coxa protraction is horizontal only, and tibia flexion lowers the foot slightly.
- Conclusion: each leg's tonic bias exceeds its rhythmic modulation. Next to check: proprioceptive phase-transition mechanisms.

## Former limitation: a global antennal-lobe / mushroom-body runaway (fixed 2026-10-03, see "Spontaneous activity, adaptation and homeostasis")

Strong, sustained bilateral sugar input (both labellar sugar sets at 100 Hz) drives the Shiu et al. spiking model into a self-sustaining global state:
- APL about 435 Hz
- antennal-lobe local neurons about 200 Hz
- mushroom-body input neurons (MBIN) about 330 Hz
- Kenyon cells about 60 Hz
- MN9 silenced

It happens in their exact model at 0.1 ms (`tools/reference_lif.py`), with or without the mirror patch (the patch delays it from 0.4 to 1.4 s), and with APL's output removed. Their published 1-second trials mostly end before it fully shows.

The ignition path, at 10 ms resolution:
1. ALIN (taste input to the antennal lobe) at 280 ms
2. antennal-lobe local neurons at 340 ms
3. the whole antennal lobe, lateral horn and mushroom body within 20 ms after that

Real antennal lobes have spike-frequency adaptation, gap junctions and non-spiking local neurons (Seki et al. 2010), and none of these is in the model. That is the next fidelity fix for the central brain.

## Ventral nerve cord (Phase 4): BANC joined to the FlyWire brain

`tools/build_vnc.py` takes the nerve cord from **BANC**, the brain-and-nerve-cord connectome (Bates et al. 2025; female, one animal; Codex release 888, `data/raw/banc/`). It joins it to the FlyWire brain through the neck, as in the real animal.

**How the two datasets are joined:**

| Neuron group | Method | Result |
|---|---|---|
| Descending neurons | name + side; within a group, paired by brain-input similarity | 1,125 / 1,299 matched |
| Ascending neurons | similarity of brain connectivity (partner cell types; BANC uses FlyWire type names for 89% of brain neurons), optimal assignment | 1,139 / 1,736 matched (median cosine 0.65) |
| Sensory-ascending neurons | same method | 128 / 581 matched |

**Checks on the matching:**
- Run blind on descending neurons, the similarity matcher picks the right type 81% of the time.
- **95%** of matched ascending pairs, and 88% of descending pairs, agree on predicted transmitter, consistently across the similarity range.

**Added nerve-cord neurons: 22,832**

| Class | Neurons |
|---|---|
| Interneurons | 12,827 |
| Sensory (leg/wing/haltere/abdomen chordotonal, campaniform, hair-plate, bristle and taste neurons) | 7,436 |
| Motor (leg, wing, haltere, neck, abdominal) | 699 |
| Unmatched ascending | 650 |

Only BANC synapses inside nerve-cord neuropils are used (818,000 connections, 7.5 million synapses); the brain stays FlyWire's. Total model size: **161,471 neurons, 15.9 million connections**. Turn the nerve cord off with `FLY_NO_VNC=1`.

**Electrical synapses** (`data/vnc/electrical_synapses.json`):
- Connectomes record chemical synapses only. The giant fibre's gap junctions onto TTMn (jump motor neuron) and PSI are documented (ShakB; Allen et al. 2006; Tanouye & Wyman 1980). They are modelled as true voltage coupling: one spike → +8 mV on the partner's membrane at the next 0.5 ms step (`lif.glsl` mode 2).
- With the giant fibre driven at about 100–115 Hz, TTMn follows at 0.73–0.85 of its spikes and PSI at 0.70–0.79. PSI drives the DLM flight-muscle motor neurons through 397 chemical synapses.
- The escape take-off is now executed when **TTMn** fires (`vnc_jump_ttmn`).

**Leg taste through the nerve cord:**
- 701 BANC leg taste neurons are placed on the tarsi of their actual leg. Modality comes from BANC receptor labels: 211 sugar (Gr5a/Gr64f), 82 sugar/low-salt, 9 bitter and 398 pheromone (ppk23/25, Ir52).
- On a sucrose patch, the leg sugar neurons fire at 30 Hz, their ascending targets at 42 Hz, and those neurons' brain targets at 7 Hz. **MN9 stays at 0**, so there is still no proboscis extension from the legs.
- Anatomically, leg sugar neurons put 8,865 of their nerve-cord synapses onto FlyWire-matched ascending neurons. Those are mostly **GABAergic** in both datasets, and so are their main brain targets (DNg103, VESa1_P02). The route to the feeding circuit appears to be **disinhibitory**.
- The Shiu et al. model has no spontaneous activity, so silencing an already-silent inhibitory neuron has no effect.

**What's needed next, all central-brain physiology rather than wiring:**
- spontaneous / resting activity;
- spike-frequency adaptation;
- non-spiking neurons.

The antennal-lobe/mushroom-body runaway (above) points to the same missing properties.

Leg motor neurons are annotated by muscle and joint action (for example `flex_femur_tibia_joint`), and so are leg proprioceptors. Connecting these to the body's joints (biomechanics) is the next body step.

Performance with everything on (brain + nerve cord + FlyVis + graded optic lobe + taste): about 25 fps, close to real time.

## License

- **Code:** Apache License 2.0 ([LICENSE](LICENSE)).
- **Documentation, figures, recordings and derived data:** CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).
- **Third-party datasets, models and assets** keep their own licenses and must be cited: see [THIRD_PARTY.md](THIRD_PARTY.md).
- **Citation:** see [CITATION.cff](CITATION.cff).
