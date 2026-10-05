# Project rules for Claude sessions (FLY-UI)

## README upkeep on every commit (owner's rule, 2026-10-05; the repo is public)
Before each commit, read `README.md` (and `HANDOFF.md` if the change touches the work plan) and update anything the
commit makes stale: numbers (neuron/edge counts, rates, benchmark results), default settings, command-line switches,
file names, "what is in it" tables, status lines and known limitations. Old results that are no longer current stay
only as clearly dated history (e.g. "2026-10-03: ..."), never presented as the current state. Keep the
"Current status" section at the top and the calibration ledger accurate.

## Fidelity rules
- Label every value by basis: measured / literature / assumed / tuned. Never present synthetic or tuned data as
  measured. Keep the calibration ledger (README) complete for anything that affects behaviour.
- Report failures with numbers; keep the README honest about what does not work yet.

## Machine and data rules
- Python: use `.venv/Scripts/python` only. Godot: `F:/GODOT4.6/Godot_v4.6-stable_win64_console.exe`.
- Godot test runs: always `--quit-after N`; on this PC also `--disable-vsync --fixed-fps 30` (or 60/120 for fine
  time resolution); never two GPU Godot runs at once; re-import (`--headless --import`) after any shader edit.
- Ask before downloads (small published data tables < 50 MB into `data/raw/` are pre-approved); record each source
  and checksum in `tools/fetch_data.py`. Raw data stays git-ignored.
- Machine-wide rules in `~/.claude/CLAUDE.md` apply (no writes outside this folder, no system changes).
