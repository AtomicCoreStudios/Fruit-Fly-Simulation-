"""Build REFERENCES.md: data/code sources (exact URLs, pinned commits, licenses) and every publication cited in the
code and docs, with title/journal/year fetched from Crossref by DOI (so links and metadata are not hand-typed).
DOIs were verified by title against Crossref on 2026-10-05. usage: build_references.py"""
import html
import json
import pathlib
import re
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOIS = {  # cited as -> DOI (published version where one exists)
    "Ache et al. 2019": "10.1016/j.cub.2019.01.079", "Allen et al. 2006": "10.1016/j.semcdb.2005.11.011",
    "Amin et al. 2020": "10.7554/eLife.56954", "Azevedo et al. 2020": "10.7554/eLife.56754",
    "Azevedo et al. 2024": "10.1038/s41586-024-07389-x", "Bates et al. 2025": "10.1101/2025.07.31.667571",
    "Bhandawat et al. 2007": "10.1038/nn1976", "Bidaye et al. 2014": "10.1126/science.1249964",
    "Bidaye et al. 2020": "10.1016/j.neuron.2020.07.032", "de Bruyne et al. 2001": "10.1016/s0896-6273(01)00289-6",
    "Dallmann et al. 2025": "10.1038/s41586-025-09554-2", "Dorkenwald et al. 2024": "10.1038/s41586-024-07558-y",
    "Eckstein et al. 2024": "10.1016/j.cell.2024.03.016", "Engert et al. 2022": "10.7554/eLife.78110",
    "Falk et al. 1976": "10.1002/jmor.1051500206", "Fisek & Wilson 2014": "10.1038/nn.3613",
    "Flood et al. 2013": "10.1038/nature12208", "Gengs et al. 2002": "10.1074/jbc.m207133200",
    "Gonzalez-Bellido et al. 2011": "10.1073/pnas.1014438108", "Gordon & Scott 2009": "10.1016/j.neuron.2008.12.033",
    "Gouwens & Wilson 2009": "10.1523/jneurosci.0764-09.2009", "Hallem & Carlson 2006": "10.1016/j.cell.2006.01.050",
    "Hige et al. 2015": "10.1038/nature15396", "Hiroi et al. 2002": "10.2108/zsj.19.1009",
    "Hiroi et al. 2004": "10.1002/neu.20063", "Hooper et al. 2009": "10.1523/jneurosci.5510-08.2009",
    "Horne et al. 2018": "10.7554/eLife.37550", "Inagaki et al. 2012": "10.1016/j.cell.2012.02.027",
    "Kazama & Wilson 2008": "10.1016/j.neuron.2008.06.015", "Kim et al. 2017": "10.7554/eLife.23386",
    "Lappalainen et al. 2024": "10.1038/s41586-024-07939-3", "Lee et al. 2025": "10.1038/s41467-025-59302-3",
    "Lobato-Rios et al. 2022": "10.1038/s41592-022-01466-7", "Maisak et al. 2013": "10.1038/nature12320",
    "Mamiya et al. 2018": "10.1016/j.neuron.2018.09.009", "Marella et al. 2012": "10.1016/j.neuron.2011.12.032",
    "Matsliah et al. 2024": "10.1038/s41586-024-07981-1", "McKellar et al. 2016": "10.7554/eLife.19892",
    "Mendes et al. 2013": "10.7554/eLife.00231", "Muench & Galizia 2016": "10.1038/srep21841",
    "Nagel & Wilson 2011": "10.1038/nn.2725", "Olsen & Wilson 2008": "10.1038/nature06864",
    "Papadopoulou et al. 2011": "10.1126/science.1201835", "Pugliese et al. 2025": "10.1101/2025.09.12.675944",
    "Root et al. 2008": "10.1016/j.neuron.2008.07.003", "Schenk & Gaudry 2023": "10.1523/eneuro.0109-22.2022",
    "Schlegel et al. 2024": "10.1038/s41586-024-07686-5", "Seki et al. 2010": "10.1152/jn.00249.2010",
    "Shiu, Sterne et al. 2022": "10.7554/eLife.79887", "Shiu et al. 2024": "10.1038/s41586-024-07763-9",
    "Sizemore et al. 2023": "10.1038/s41467-023-41012-3", "Sterne et al. 2021": "10.7554/eLife.71679",
    "Stuerner et al. 2025": "10.1038/s41586-025-08925-z", "Tanouye & Wyman 1980": "10.1152/jn.1980.44.2.405",
    "Tastekin et al. 2026": "10.1016/j.cell.2026.08.016", "Tastekin et al. 2025 (preprint)": "10.1101/2025.08.25.671814",
    "Thorne et al. 2004": "10.1016/j.cub.2004.05.019", "Turner et al. 2008": "10.1152/jn.01283.2007",
    "Todorov et al. 2012": "10.1109/IROS.2012.6386109", "Tuthill & Wilson 2016": "10.1016/j.cub.2016.06.070", "von Reyn et al. 2014": "10.1038/nn.3741",
    "Wang et al. 2004": "10.1016/j.cell.2004.06.011", "Wang-Chen et al. 2024": "10.1038/s41592-024-02497-y",
    "Weiss et al. 2011": "10.1016/j.neuron.2011.01.001", "Wilson et al. 2004": "10.1126/science.1090782",
    "Zhao et al. 2025": "10.1038/s41586-025-09276-5", "Zheng et al. 2018": "10.1016/j.cell.2018.06.019",
    "Zill et al. 2004": "10.1016/j.asd.2004.05.005",
}

SOURCES = """## Data and code sources

Raw downloads go to `data/raw/` (not committed). `tools/fetch_data.py` records URLs, pinned commits and checksums.

| Source | What we use | Link | License / terms |
|---|---|---|---|
| FlyWire v783 connectome, as exported by Shiu et al. | brain wiring (`Completeness_783.csv`, `Connectivity_783.parquet`), reference LIF model | https://github.com/philshiu/Drosophila_brain_model | MIT; FlyWire citation guidelines |
| FlyWire cell-type annotations (Schlegel et al. 2024) | cell types, transmitters, sides, positions | https://github.com/flyconnectome/flywire_annotations | see repository |
| FlyWire Codex v783 data | visual columns and types, cell statistics, community labels | https://codex.flywire.ai | FlyWire citation guidelines (see Codex) |
| BANC brain-and-nerve-cord connectome (Bates et al. 2025) | nerve-cord wiring and annotations (v626), skeleton cable lengths | Dataverse: https://doi.org/10.7910/DVN/8TFGGB · Codex: https://codex.flywire.ai/banc · portal: https://banc.community | CC BY 4.0 |
| Neck connective tables (Stuerner et al. 2025) | DN/AN types for joining brain and nerve cord | https://github.com/flyconnectome/2023neckconnective | see repository |
| Pugliese et al. VNC CPG model | rate-model code and parameters, BANC front-leg subnetwork, surface areas | https://github.com/smpuglie/Pugliese_2026 (commit 10e7661) · https://zenodo.org/records/22260924 | data CC BY 4.0; code see repository |
| Tastekin et al. gustatory connectome | taste/leg GRN types, feeding motor neurons (Suppl. Table 2) | https://doi.org/10.1101/2025.08.25.671814 · https://doi.org/10.1016/j.cell.2026.08.016 | see publication |
| DoOR 2.0 odorant response database | ORN spontaneous rates | https://github.com/Dahaniel/DoOR.data | see repository |
| Micro-CT eye map (Zhao et al. 2025) | 1,709 ommatidia positions and directions | https://github.com/reiserlab/eyemap_T4 (commit 99d2a43) | GPL-3.0 |
| NeuroMechFly v2 / FlyGym | body meshes, rig, flybody and musculoskeletal leg models (`assets/flygym`) | https://github.com/NeLy-EPFL/flygym (commit 38c8ec6) | Apache 2.0 |
| FlyVis | graded optic lobe model (flow/0000/000) | https://github.com/TuragaLab/flyvis | see repository |
| Virtual Fly Brain CATMAID (FAFB) | skeleton soma positions for cross-checks | https://fafb.catmaid.virtualflybrain.org | see VFB terms |
| MuJoCo | physics engine (musculoskeletal validation) | https://github.com/google-deepmind/mujoco | Apache 2.0 |

"""


def meta(doi):
    for _ in range(4):
        try:
            url = f"https://api.crossref.org/works/{doi}?mailto=info@atomiccorestudios.com"
            m = json.load(urllib.request.urlopen(url, timeout=30))["message"]
            t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(m["title"][0]))).strip()
            j = html.unescape((m.get("container-title") or ["bioRxiv"])[0] or "bioRxiv")
            return t, j, m["issued"]["date-parts"][0][0]
        except Exception:
            time.sleep(5)
    return "", "", ""


def main():
    rows = []
    for key in sorted(DOIS, key=lambda k: re.sub(r"^(von|de) ", "", k).lower()):
        t, j, y = meta(DOIS[key])
        time.sleep(0.4)
        rows.append(f"- **{key}.** {t}. *{j}* ({y}). https://doi.org/{DOIS[key]}")
    txt = ("# References\n\nAll data, code and publications used by FLY-UI. Licenses for redistribution: see "
           "[THIRD_PARTY.md](THIRD_PARTY.md). Generated by `tools/build_references.py` (DOIs verified against "
           "Crossref).\n\n" + SOURCES + "## Publications cited in the code and documentation\n\n" + "\n".join(rows) + "\n")
    (ROOT / "REFERENCES.md").write_text(txt, encoding="utf-8")
    print(len(rows), "publications;", sum(1 for r in rows if ".** ." in r), "without title")


if __name__ == "__main__":
    main()
