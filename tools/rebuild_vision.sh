#!/usr/bin/env bash
# Rebuild the whole vision chain for one FlyVis ensemble member, e.g.:
#   bash tools/rebuild_vision.sh flow/0000/000
# Order matters: FlyVis export -> lattice on the eyes -> FlyVis->FlyWire map -> graded network
# -> graded GPU export (+ silenced LIF weights) -> offline validation figure.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-flow/0000/000}"
PY=.venv/Scripts/python
$PY tools/flyvis/export_flyvis.py --model "$MODEL"
$PY tools/flyvis/map_eye_lattice.py
[ -f data/extra_columns.csv ] || $PY tools/flyvis/assign_columns.py
$PY tools/flyvis/export_godot.py
$PY tools/graded/build_graded.py
$PY tools/graded/export_godot.py
$PY tools/graded/validate.py --plot "screenshots/graded_validation_${MODEL//\//_}.png"
