"""Export the offline VNC rate network (tools/vnc_rate_model.py setup: Pugliese et al. 2025 equations on our BANC
graph, measured/cable-estimated sizes, full nerve cord) to data/vnc/vnc_rate_net.npz, so the FlyGym environment
(.venv-flygym, no pandas) can run it in closed loop with the MuJoCo body (tools/nmf_closed_loop.py).
usage: .venv/Scripts/python tools/export_vnc_rate_net.py"""
import os, sys, pathlib
import numpy as np
os.environ.setdefault("RM_SUBNET", "full"); os.environ.setdefault("RM_SIZE", "measured")
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "tools/vnc_rate_model.py").read_text(encoding="utf-8")
g = {"__file__": str(ROOT / "tools/vnc_rate_model.py")}; argv = sys.argv; sys.argv = ["x"]
exec(compile(src[:src.index("# stimulus:")], "vnc_rate_model_setup", "exec"), g); sys.argv = argv
W = g["W"].tocsr()
out = ROOT / "data/vnc/vnc_rate_net.npz"
np.savez_compressed(out, ids=g["ids"].astype(np.int64), a=g["a"], theta=g["theta"], W_data=W.data.astype(np.float32),
                    W_indices=W.indices, W_indptr=W.indptr, fcap=g["fcap"], tau=g["tau"],
                    subnet=os.environ["RM_SUBNET"], size=os.environ["RM_SIZE"])
print(f"{g['n']} neurons, {W.nnz} edges -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
