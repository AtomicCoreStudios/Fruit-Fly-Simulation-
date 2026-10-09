"""Download every external dataset this project uses, from its original publisher, pinned by
SHA-256 (the exact files the results in README were produced with). Raw data is not stored in git.

usage: .venv/Scripts/python tools/fetch_data.py            (skips files that are already present)
then:  rebuild derived data, see README "Rebuild from a fresh clone".
"""
import hashlib, pathlib, subprocess, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
GH = "https://raw.githubusercontent.com"
EYEMAP_COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"

FILES = [
    # FlyWire v783 connectome as exported by Shiu et al. 2024 (Nature 634:210)
    (f"{GH}/philshiu/Drosophila_brain_model/main/Completeness_783.csv", "Completeness_783.csv",
     "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    (f"{GH}/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet", "Connectivity_783.parquet",
     "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    (f"{GH}/philshiu/Drosophila_brain_model/main/LICENSE", "LICENSE", None),
    # FlyWire cell-type annotations, Schlegel et al. 2024 (Nature 634:139)
    (f"{GH}/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
     "annot_Supplemental_file1_neuron_annotations.tsv",
     "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd"),
    # FlyWire Codex v783 (Matsliah et al. 2024, Nature 634:166): visual columns and types
    ("https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz",
     "column_assignment.csv.gz", "bdf4ce7f62cc63493d53eefad3816ff2dfd08b190e97b35a492e0e453df2f0f6"),
    ("https://storage.googleapis.com/flywire-data/codex/data/fafb/783/visual_neuron_types.csv.gz",
     "visual_neuron_types.csv.gz", "4bcc6a2f98b86e6c3fb7eaddb49736f3d81ab65bda35da8f740641201a1e379f"),
]
# Zhao et al. 2025 (Nature): micro-CT eye map, reiserlab/eyemap_T4 (GPL-3), pinned commit
EYEMAP = ["LICENSE", "data/eyemap.RData", "data/lens_ixy.RData", "data/med_ixy.RData", "data/Mi1_ind.RData",
          "data/hexnb_ind_dist.RData", "data/chiasm.RData", "data/microCT/20231107.RData",
          "data/microCT/20240530.RData", "data/microCT/20240701.RData", "data/microCT/20240701_dia.RData",
          "data/microCT/20240701_nb.RData", "data/microCT/20240701_normals.RData",
          "data/microCT/20240701_position/cone.csv", "data/microCT/20240701_position/lens.csv",
          "data/microCT/20240701_roc.RData"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(url, dest, digest=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("download", url)
        urllib.request.urlretrieve(url, dest)
    if digest and sha(dest) != digest:
        sys.exit(f"checksum mismatch for {dest} - the upstream file changed; results may differ")


def main():
    for url, name, digest in FILES:
        get(url, RAW / name, digest)
    for p in EYEMAP:
        out = RAW / "eyemap_T4" / (p if p == "LICENSE" else p.replace("data/", "", 1))
        get(f"{GH}/reiserlab/eyemap_T4/{EYEMAP_COMMIT}/{p}", out)
    (RAW / "eyemap_T4/SOURCE_COMMIT.txt").write_text(EYEMAP_COMMIT + "\n")
    for d in (RAW, RAW / "eyemap_T4"):
        (d / ".gdignore").touch()
    # FlyVis pretrained ensemble (Lappalainen et al. 2024), via the official FlyVis downloader
    if not (ROOT / "assets/flyvis/results/flow/0000/000").exists():
        env = {**__import__("os").environ, "FLYVIS_ROOT_DIR": str(ROOT / "assets/flyvis")}
        exe = pathlib.Path(sys.executable).with_name("flyvis.exe" if sys.platform == "win32" else "flyvis")
        subprocess.run([str(exe), "download-pretrained"], check=True, env=env)
        (ROOT / "assets/flyvis/.gdignore").touch()
    print("all data present and verified")


if __name__ == "__main__":
    main()


# Pugliese et al. 2025/2026 VNC CPG model data (CC-BY-4.0), fetched 2026-10-05 (user approved the 157 MB file):
#   data/raw/pugliese/DNg100_Stim_Prune_BANC_vncOnly.zip  Zenodo 22260924, md5 b73648ba5bfce8e28739816d4c28fc0b,
#     sha256 5f509a46a385e4b56c36142622ae828f985ee85c80bc2d3cb10d235f3474ee55
#   from github.com/smpuglie/Pugliese_2026 @10e7661, data/banc t1 premotor/:
#     W_20260217.npz                                   sha256 83197529fa336b5f9ce400cf689f299fa3e8f6f5379947ae7df5ec004869141f
#     wTable_20260217_fullData_consistentColumns.csv   sha256 6f87bf62227e160754523418bfbac35fc8971a90e4ed54574bf769b08332e285
#     wTable_20260217_surfAreas.csv                    sha256 07c71780f861f807a620330e35bc2a57176dce76110894b6082ea1aca7b545b0
PUGLIESE = {
    "zenodo_record": "https://zenodo.org/records/22260924",
    "github": "https://github.com/smpuglie/Pugliese_2026/tree/10e7661",
}

# BANC skeletons (Bates et al. 2025, Harvard Dataverse doi:10.7910/DVN/8TFGGB, file id 11845086, CC-BY-4.0),
# fetched 2026-10-05 (user approved, 215.6 MB): data/raw/banc/neuron_skeletons.zip md5 f7d16e8ed00454ce26c3c37a351df004
BANC_SKELETONS = "https://dataverse.harvard.edu/api/access/datafile/11845086"

# MANC v1.0 neuron properties (Janelia FlyEM, Takemura et al. 2024 / Marin et al. 2024 / Cheong et al. 2024; CC-BY-4.0),
# public export bucket gs://flyem-manc-exports, fetched 2026-10-09 (17.2 MB, pre-approved small table):
# data/raw/manc/manc-v1.0-neuron-properties.feather
#   sha256 0c4476528906bb0a20e05e1f01e83fc2e5a582761536171ede5463669ca0b891
# Used to check leg campaniform-sensillum annotations (tools/manc_leg_sensory_check.py).
MANC_PROPERTIES = "https://storage.googleapis.com/flyem-manc-exports/v1.0/manc-v1.0-neuron-properties.feather"
