#!/usr/bin/env bash
# resting-state test: tethered, no vision, 10 s; prints class rates vs literature targets
set -e; cd "$(dirname "$0")/.."; TAG=${1:-rest}
"/f/GODOT4.6/Godot_v4.6-stable_win64_console.exe" --path . --disable-vsync --fixed-fps 30 --quit-after 2600 -- --tethered --vision=none --frames=600 --clean_air --record=recordings/rest_$TAG.csv > /dev/null 2>&1
.venv/Scripts/python - "$TAG" <<'PY'
import sys, pandas as pd
d=pd.read_csv(f'recordings/rest_{sys.argv[1]}.csv'); d=d[d.t_ms>2000]
want={'sensory:olfactory':10,'central:ALPN':4,'central:ALLN':2,'central:Kenyon_Cell':0.2,'central:MBON':6,'central:DAN':2,'central:LHLN':3,'central:LHCENT':3,'type:APL':0,'type:MN9':0.5}
row=[]
for k,t in want.items():
    cols=[c for c in d.columns if c.startswith(k+'_')]
    row.append(f"{k.split(':')[1]} {d[cols].mean().mean():.1f}/{t}")
print(sys.argv[1], ' | '.join(row))
PY
