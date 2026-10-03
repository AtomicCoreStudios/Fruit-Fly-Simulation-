# Handoff: FLY-UI, the next phase: anatomically faithful sense organs, built in Blender

You are continuing an existing project in `F:\Fruit Fly Experiment`. Read this whole file before acting.
Then read `README.md` and skim `scripts/fly.gd`, `scripts/fly_brain.gd`, `shaders/lif.glsl` and
`tools/build_connectome.py`.

## 1. Research goal

The user is testing the "UI" (Uploaded Intelligence) idea from the TV show *Pantheon* on the fruit fly,
*Drosophila melanogaster*. The pipeline is: real connectome → spiking emulation → embodied in a virtual
world in real time. The aim for this phase is a **1:1 body**. Every sense organ should match the real
anatomy (positions, counts, optical axes and receptor types). Each receptor should drive the **specific
real FlyWire neurons** it connects to, so that brain function can be studied inside the world.

Rules for fidelity:
- **Measured data beats guesses.** Where a real dataset exists, use it and cite it.
- **Label every approximation.** State it in the code comment, in `data/sensor_map.json` (field
  `"basis"`: `measured` | `literature` | `approximate`) and in the README.
- **Never present synthetic or approximate data as real.** A true 1:1 is limited by the available data.
  Say where it breaks: see section 7.

## 2. Machine rules (also in `C:\Users\artez\.claude\CLAUDE.md`, which is enforced by hooks)

- **Where you may write:** only inside `F:\Fruit Fly Experiment`, the scratchpad and `~/.claude`. No
  registry edits, system settings, admin rights or installers. Ask before anything destructive or
  outside the project.
- **Deleting files:** the path-guard hook misreads paths containing spaces in `Remove-Item`. Overwrite
  files instead of deleting them, or ask the user.
- **Godot 4.6:** `F:\GODOT4.6\Godot_v4.6-stable_win64_console.exe`. Always pass `--quit-after N` on test
  runs; a parse error once left a window hanging. Re-import after adding assets:
  `--headless --path . --import`.
- **Blender:** `F:\Blender\blender.exe`, controlled through the **Blender MCP** server (just set up;
  registered with `--scope user`, port 9876). The user must have Blender open with **Connect to Claude**
  clicked in the N-sidebar MCP tab. The `execute_blender_code` tool runs arbitrary Python, so save
  `.blend` files into `F:\Fruit Fly Experiment\blender\` often.
- **Python:** use the project environment, `.venv\Scripts\python` (numpy, pandas, pyarrow). Don't pip
  install into the system Python.

## 3. What already exists and works

| Part | File(s) | State |
|---|---|---|
| Connectome builder | `tools/build_connectome.py` | `flywire` mode: the real FlyWire v783 connectome, **138,639 neurons, 15,091,983 connections, 54.5M synapses**. `synthetic` mode: the hand-designed stand-in. Output: `data/connectome_{mode}.bin` and `.json` |
| GPU brain | `shaders/lif.glsl`, `scripts/fly_brain.gd` | See the note below this table |
| Body and senses | `scripts/fly.gd` | A **placeholder** procedural body built from spheres and boxes. Senses are crude: 6 eye sectors, 2 antenna odour points, a wind rule and patch-based taste |
| World, HUD, viewer | `scripts/main.gd`, `shaders/neuron_view.gdshader` | See the note below this table |
| Raw data | `data/raw/` | The Shiu et al. repo files and the FlyWire annotations (`annot_Supplemental_file1..5`). The annotations give cell type, class, side, soma position and neurotransmitter for every neuron |

**GPU brain:** a leaky integrate-and-fire (LIF) model with Shiu et al. 2024 parameters: 0.275 mV per
synapse, 1.8 ms delay, input reset on spike, and 68.75 mV Poisson inputs. It uses a 0.5 ms step, runs in
real time at 60 fps on the RTX 2080, and keeps per-population spike counters. **Inputs are per
group:** a Poisson rate for each group name. For per-ommatidium vision you will need per-neuron input
rates; add an optional per-neuron rate buffer to the shader.

**World, HUD and viewer:** an arena with food and bitter patches, pillars, a looming predator, wind and
the sun. The HUD shows region, sensory and motor bars. There is a live 3D connectome viewer.

**Test harness:** `--log`, `--pops`, `--auto`, `--threat_at=N`, `--start=x,z`, `--yaw=a`, `--frames=N`,
`--shot=path.png`, `--record=path.csv`, `--connectome=flywire|synthetic`, `--opto`.

**Findings so far with the real brain:**
- It does not walk spontaneously. Optogenetic drive of DNp09 (key **O**) makes it walk.
- DNa02_L fires tonically and DNa02_R is silent, so the fly circles left.
- There is **no looming escape**: LPLC2 and the giant fibre (DNp01) stay silent. The likely cause is
  that the eye input is too crude to create the T4/T5 motion signals loom detection needs. Real
  per-ommatidium vision is the fix this phase targets.
- APL fires at over 300 Hz; the real APL is graded and non-spiking.
- The EPG compass neurons are silent.

## 4. Assets to use: NeuroMechFly / FlyGym (Apache-2.0, NeLy-EPFL, github.com/NeLy-EPFL/flygym)

The FlyGym body comes from a micro-CT scan of a real adult female fly. It has separate STL meshes per
segment in `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/`, for example:
- `c_head`, `c_thorax`, `c_abdomen*`, `c_haustellum`, `c_rostrum`
- `l_eye`, `l_arista`, `l_funiculus`, `l_pedicel`, `l_wing`, `l_haltere`
- leg segments, e.g. `lf_coxa`, `lf_trochanterfemur`, `lf_tibia`, `lf_tarsus1-5`, and the same for the
  other legs

It also has the MuJoCo joint model (MJCF XML) in `src/flygym/assets/model/neuromechfly/`, and a second
model, `flybody/fruitfly.xml` (Janelia/DeepMind).

Their eye model uses **721 ommatidia per eye on a hexagonal grid, 70% yellow / 30% pale**. It was coupled
to FlyVis, the connectome-constrained visual-system model of Lappalainen et al. (Nature 2024).

The user said to move forward with the organs, so downloading these assets is approved. Download them
into `assets/flygym/` and keep the LICENSE file. Before downloading, state the file count and size in
one line. Check whether meshes exist for the right-hand side or need mirroring.

## 5. The work plan (do it in order; show the user progress with screenshots)

**Phase 0 — Check the setup.**
- Call a Blender MCP tool to list the scene. If it fails, walk the user through step 5 of the MCP setup:
  open Blender, press N, open the MCP tab, click Connect to Claude, then start a new session.
- Download the FlyGym assets (section 4).

**Phase 1 — Assemble the body in Blender.**
- Import the STLs and place each segment using the MJCF body frames, keeping real scale (millimetres).
  Build an armature from the MJCF joint tree.
- Give it plausible materials: red-brown eyes, translucent wings.
- Save as `blender/fly_body.blend`. Screenshot for the user.

**Phase 2 — Compound eyes, the priority.**
- Fit about 721 ommatidia per eye as a hexagonal lattice on the `l_eye` surface, and mirror for the
  right eye. Each facet gets:
  - a position and an optical axis (the surface normal, corrected to the known eye map where possible;
    the interommatidial angle is about 4.5–5°)
  - an acceptance angle (about 5°)
  - a type: yellow or pale, plus a dorsal-rim-area (DRA) row for polarized light
- Export `data/ommatidia.json` with: id, side, row/column (hex coordinates), position, direction, type,
  and FlyWire column neurons.
- **Map each facet to FlyWire columns.** Research the medulla column map published with FlyWire
  (Matsliah et al. 2024, *Nature*, the optic-lobe "parts list", and its data or code release). Each
  column's R7, R8, L1, L2, L3, Mi1, Tm3 and so on come from that map. Remember the lamina is upright and
  the medulla is inverted by the optic chiasm. Note that R1–6 are largely not reconstructed, so drive
  their postsynaptic lamina cells.
- If no published column map exists, build one from neuron positions and label it `approximate`.

**Phase 3 — The other organs.** One sensor point (an empty) per receptor site, with FlyWire neuron IDs:
- **Antenna:** pedicel, funiculus (third segment) and arista. Johnston's organ reads how far wind
  rotates the arista and funiculus. Split its neurons into subgroups for wind/gravity versus hearing.
  Place olfactory sensilla on the funiculus by type: basiconic, trichoid, coeloconic.
- **Olfactory receptors:** map each receptor-neuron type from the FlyWire annotations to its sensillum
  class. Include the **maxillary palp** receptor types.
- **Proboscis:** labellar taste sensilla and taste pegs, and the MN9 output (proboscis extension).
- **Legs:** taste and mechanosensory sensilla on the 5 foot segments, detected by contact.
- **Other:** 3 ocelli; halteres; wing hinge; neck and eye bristles (the annotations have "eye bristle"
  and "head bristle" mechanosensory classes).
- Also wire the input groups that already exist for temperature, humidity and water taste.

**Phase 4 — Export and integrate into Godot.**
- Export the body as a glTF file (`assets/fly/fly.glb`) with named empties for every sensor, plus
  `data/sensor_map.json` mapping each sensor to FlyWire root IDs.
- In Godot, replace the procedural body in `fly.gd`.
- **Vision:** render from the head, either a cubemap or two wide field-of-view cameras. Sample each
  ommatidium's cone on the GPU, write per-neuron rates straight into the brain's input buffer, and keep
  it real time.

**Phase 5 — Validate against real behaviour.**
- Behaviours to reproduce: loom → LPLC2 → giant-fibre escape; the optomotor response; phototaxis;
  proboscis extension to sugar.
- Record CSVs of each test.
- Report honestly which behaviours emerge from the real wiring and which don't.

## 6. How to work with this user

- The user is non-technical-to-intermediate and driven by the research question.
- Keep status updates short and in plain words, and send screenshots of progress (SendUserFile).
- Ask before downloads outside what's already approved, and before anything outside the project folder.
- When something fails, say so plainly and show the evidence.

## 7. Known limits on "1:1" (state them; don't hide them)

**Missing data:**
- FlyWire covers the head brain only. There is **no nerve cord** (VNC): see MANC, FANC and BANC as a
  future step.
- R1–6 photoreceptors, and parts of the lamina, are incomplete.

**Model limits:**
- LIF neurons have no graded or non-spiking cells (APL, many optic-lobe neurons are graded), no
  neuromodulators, no plasticity, no gap junctions and no individual brain state.
- The body mapping from descending neurons to movement is still a simple calibration. FlyGym's walking
  controllers (central pattern generators), which run on a nerve-cord abstraction, are a possible
  upgrade.

## 8. Progress log

**2026-09-30:**
- **Phase 0 done:**
  - Blender 5.1.2 connected over MCP.
  - FlyGym assets are in `assets/flygym/` (commit in `SOURCE_COMMIT.txt`). Only left meshes exist, so the right side is mirrored as FlyGym does.
- **Phase 1 done:**
  - `tools/blender/build_body.py` builds the body. It is an Empty-per-body hierarchy using the exact MuJoCo frames, 1 unit = 1 mm, with `apply_pose()` for FlyGym pose yamls.
  - No armature was built. Joints are the Empty origins, which export cleanly to Godot nodes.
- **Phase 2 mostly done:**
  - `tools/build_eye_map.py` → `data/ommatidia.json`, and `tools/blender/place_ommatidia.py` → facets on the eyes plus `data/ommatidia_placement.json`.
  - Extra downloads (user approved): `data/raw/column_assignment.csv.gz`, `data/raw/visual_neuron_types.csv.gz`, `data/raw/eyemap_T4/`, and `rdata` pip-installed into `.venv`.
  - Correction to section 7: R1–6 *are* in FlyWire v783 (7,172, typed in Codex) and connect to L1–L3.
- **Next:**
  - Phase 4 vision first. Sample each facet's cone in Godot and write per-neuron Poisson rates (add the per-neuron rate buffer to `lif.glsl`). This is the direct test of the loom → LPLC2 → giant-fibre hypothesis.
  - Then Phase 3 organs.
- **Phase 4, eyes, done (2026-09-30):**
  - `assets/fly/fly.glb` body in Godot (`fly.gd _load_body`), facet vision (`scripts/fly_eye.gd`, `shaders/eye.glsl`), per-neuron input buffer in `lif.glsl`, photoreceptors set to histamine/inhibitory.
  - `.gdignore` added to `blender/`, `recordings/`, `data/raw/`, `assets/flygym/`. Godot was importing CSVs as translations and hung.
- **Phase 5, loom, first result:**
  - No escape. The signal reaches T5 with correct polarity only when columnar cells get a sub-threshold resting bias. It dies before LPLC2.
  - See the README "Loom test" section.
- **Next:**
  - Graded optic-lobe model (FlyVis parameters) or per-type time constants.
  - Then Phase 3 organs, attached to the named body nodes in `fly.glb`.
- **FlyVis hybrid (2026-09-30):**
  - Installed in `.venv` (user approved): torch CPU and flyvis (`--ignore-requires-python`; FlyVis caps Python below 3.13).
  - Pretrained models are in `assets/flyvis/` (9.4 MB, gdignored).
  - Tools are in `tools/flyvis/`, the shader is `shaders/flyvis.glsl`, and `optic_lobe=flyvis` is now the default.
  - Result: loom-specific T4/T5 responses appear, but LPLC2 stays silent and the giant fibre fires only false alarms (README).
- Also lowered the per-frame step cap in `fly_brain.gd` from 50 to 34 ms. The old cap could lock the loop at about 19 fps.
- **Graded optic lobe beyond FlyVis and two-compartment VPNs (2026-10-01):**
  - Code: `tools/graded/*`, `shaders/graded.glsl`, `tools/flyvis/assign_columns.py`, `screen_ensemble.py`, `pd_check.py`, and `tools/rebuild_vision.sh`.
  - Default FlyVis model: flow/0000/000.
  - Result: LPLC2 is loom-specific and the wiring is radially correct. LC4 has ON-cell polarity inherited from FlyVis T2. No clean escape. Details in the README.
  - Performance: about 25 fps, roughly 0.85× real time with everything on.
