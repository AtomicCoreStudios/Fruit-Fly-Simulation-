"""Render the fly's-eye view from a --eye_dump CSV: each ommatidium is drawn as a disc at its
measured optical axis (equirectangular map, azimuth left->right = fly's left ... right,
elevation up), grey level = the luminance that facet sampled. No plotting libraries needed.

usage: .venv/Scripts/python tools/plot_eye_view.py recordings/eye_dump.csv screenshots/eye_view.png
"""
import json, sys, zlib, struct, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = 2  # pixels per degree


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def panel(dirs, val, lo, hi):
    img = np.full((180 * S, 360 * S, 3), 40, np.uint8)
    az = np.degrees(np.arctan2(-dirs[:, 2], dirs[:, 0]))   # + = left
    el = np.degrees(np.arcsin(np.clip(dirs[:, 1], -1, 1)))
    yy, xx = np.mgrid[-6:7, -6:7]
    disc = (xx ** 2 + yy ** 2) <= 4.8 ** 2
    for a, e, v in zip(az, el, val):
        cx, cy = int((180 - a) * S), int((90 - e) * S)      # fly's left drawn on the left
        g = int(np.clip((v - lo) / (hi - lo), 0, 1) * 255)
        ys, xs = cy + yy[disc], cx + xx[disc]
        ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
        img[ys[ok], xs[ok]] = (g, g, int(g * 0.85))
    img[:, 180 * S] = (200, 60, 60)          # straight ahead
    img[90 * S, :] = (90, 90, 90)            # horizon
    return img


def main(src, dst):
    drive = json.loads((ROOT / "data/eye_drive.json").read_text())
    dirs = np.array(drive["facet_dir"])
    rows = [l.strip().split(",") for l in open(src)]
    t = np.array([float(r[0]) for r in rows]); vis = np.array([int(r[1]) for r in rows])
    lum = np.array([[float(x.split(";")[0]) for x in r[2:]] for r in rows])
    t0 = t[np.argmax(vis == 1)] if vis.any() else t[len(t) // 2]
    last = t[vis == 1].max() if vis.any() else t[-1]
    picks = [t0 - 300, t0 + 0.5 * (last - t0), t0 + 0.85 * (last - t0), last]
    lo, hi = np.percentile(lum, 1), np.percentile(lum, 99)
    panels = []
    for tp in picks:
        i = int(np.argmin(np.abs(t - tp)))
        panels.append(panel(dirs, lum[i], lo, hi))
        print(f"panel t={t[i]/1000:.2f}s (launch {t0/1000:.2f}s), min facet lum {lum[i].min():.3f}")
    sep = np.full((6, panels[0].shape[1], 3), 255, np.uint8)
    png(dst, np.concatenate(sum([[p, sep] for p in panels], [])[:-1]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
