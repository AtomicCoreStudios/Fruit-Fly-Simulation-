"""Shared setup for FlyVis tools: data root inside the project, and a Windows fix for datamate.

datamate.io._write_h5 opens an HDF5 file, and on the expected KeyError tries to delete it while its
own handle is still open, which fails on Windows (WinError 32). Patched here at runtime only."""
import os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("FLYVIS_ROOT_DIR", str(ROOT / "assets" / "flyvis"))

import numpy as np, h5py as h5
import datamate.io as _dio


def _write_h5_fixed(path, val):
    val = np.asarray(val)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()
    with h5.File(path, "w", libver="latest") as f:
        f["data"] = val
        f.swmr_mode = True


_dio._write_h5 = _write_h5_fixed
import datamate.directory as _dd  # noqa: E402
if hasattr(_dd, "_write_h5"):
    _dd._write_h5 = _write_h5_fixed
