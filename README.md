# AI4UrbanScience (AI4US)

[![CI](https://github.com/24kchengYe/AI4UrbanScience/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/24kchengYe/AI4UrbanScience/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/Citation-CFF-blue.svg)](CITATION.cff)

**AI4UrbanScience (AI4US)** is an empirically grounded framework for testing whether GenAI-generated urban data recover known urban regularities and support repeatable controlled interventions across text and image modalities.

This repository accompanies *GenAI Models Capture Urban Science but Oversimplify Complexity*. Across urban scaling, within-city distance decay, neighbourhood vitality and streetscape perception, the study asks three questions: whether a recognizable relationship is present, whether generated magnitudes and variation resemble empirical observations, and whether a model responds consistently when one urban condition is changed.

The public release contains quantitative source tables, identifier-free parsed outputs, exact submitted Figure 1–5 PDFs, scripted diagnostic rerenders for Figures 2–5 and Supplementary Figures S1–S12, and a standalone numerical recomputation for the prespecified Qwen2.5-VL-7B-Instruct Perception replication. The study's primary inspectable-model analysis uses Qwen2.5-VL-72B-Instruct-AWQ; for that analysis, this repository releases processed estimands and figure-source tables rather than raw inference, hidden states or model weights.

## Study at a glance

| Case | Urban scale | Generated or evaluated object | Empirical reference | Controlled test |
|---|---|---|---|---|
| Urban Scaling | City systems | Population, road and GDP tables | Published scaling relationships and observed city data | Change Population within fixed metropolitan descriptions |
| Distance Decay | Within-city structure | Land-density profiles | Empirical radial profiles from multiple cities | Change distance from the city centre |
| Jacobs-informed Vitality | Neighbourhoods | Morphology and pedestrian-activity tables | Observed pedestrian activity at 52 Melbourne sites | Change Functional Mix while holding four morphology indicators fixed |
| Place Pulse Perception | Streetscapes | Six-dimensional perception scores for generated and edited scenes | Place Pulse human choices | Compare Natural, Traffic and Built edits with same-scene editing controls |

Across the four cases, generated outputs recovered recognizable urban relationships and produced repeatable directional responses to controlled changes. Empirical comparison also revealed systematic compression of numerical coverage, local variation and visual diversity. In the primary Qwen2.5-VL-72B analysis, selected relation distinctions were readable from intermediate activations and some activation edits shifted output scores, while other prespecified table-level statistics remained unchanged.

## Model roles

| Model | Role in the study | Public support in this repository |
|---|---|---|
| GPT-4o | Numerical synthesis and intervention, empirical-site ranking, and streetscape-perception scoring | Released source tables and identifier-free parsed outputs; raw API replay is not provided |
| `gpt-image-2` | Streetscape generation and editing | Released metrics and source tables; source images and raw calls are not redistributed |
| Qwen2.5-VL-72B-Instruct-AWQ | **Primary inspectable-model analysis:** matched response profiles, linear readouts and activation edits | Released processed estimands, figure-source tables and the exact Figure 5 reference PDF; no weights, hidden-state tensors or raw inference |
| Qwen2.5-VL-7B-Instruct | **Prespecified Perception replication only** | Standalone numerical recomputation from released behaviour and state-replacement matrices; this does not rerun model inference |

## Quick start

The project metadata declares Python 3.10 or later. Continuous integration runs on Python 3.11, and the frozen Windows dependency snapshot was additionally validated with CPython 3.13.3. The release verifier intentionally rejects virtual environments, caches, build products and rendered outputs inside the repository, so create the environment and output directories beside the clone.

```bash
git clone https://github.com/24kchengYe/AI4UrbanScience.git
cd AI4UrbanScience
```

On Windows PowerShell:

```powershell
python -m venv ..\ai4us-venv
$ai4usPython = (Resolve-Path '..\ai4us-venv\Scripts\python.exe').Path
& $ai4usPython -m pip install -r requirements/base.txt 'pytest>=8,<9'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path 'src').Path
& $ai4usPython scripts/build_manifest.py --check
& $ai4usPython scripts/verify_release.py
& $ai4usPython -m pytest -q -p no:cacheprovider
& $ai4usPython scripts/verify_release.py
```

On macOS or Linux:

```bash
python -m venv ../ai4us-venv
AI4US_PYTHON=../ai4us-venv/bin/python
"$AI4US_PYTHON" -m pip install -r requirements/base.txt 'pytest>=8,<9'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
"$AI4US_PYTHON" scripts/build_manifest.py --check
"$AI4US_PYTHON" scripts/verify_release.py
"$AI4US_PYTHON" -m pytest -q -p no:cacheprovider
"$AI4US_PYTHON" scripts/verify_release.py
```

For the exact dependency versions used in the clean Windows validation with CPython 3.13.3, install `requirements/locked.txt` instead of `requirements/base.txt` plus pytest.

The examples below use `python`; run them with the same external interpreter (or activate that external environment). Output paths are placed beside the clone to preserve the clean-checkout verifier contract.

Render a smoke subset or the complete released-source set into a new output directory:

```bash
python scripts/reproduce_figures.py --figures 4,5,S10 --output-dir ../ai4us-outputs/smoke
python scripts/reproduce_figures.py --all --output-dir ../ai4us-outputs/all_figures
```

Recompute the released within-family 7B Perception state-replacement statistics:

```bash
python scripts/recompute_perception_state_replacement.py
```

Validate all 16 quantitative workbook sheets against their frozen CSV aliases:

```bash
python scripts/extract_source_data.py --report ../ai4us-outputs/source_data_validation.json
```

## What this release contains

| Path | Contents |
|---|---|
| `data/publication/` | Source Data and Supplementary Data 1–2 workbooks |
| `data/processed/` | Identifier-free parsed outputs released for the supported analyses |
| `data/figure_sources/` | Frozen plotted-source tables and reader-facing manifests |
| `reference_outputs/main/` | Exact submitted Figure 1–5 PDFs |
| `scripts/` | Stable validation, rerendering and numerical-recomputation entry points |
| `src/ai4us/` | Reusable analysis, validation and plotting modules |
| `provenance/` | Public-safe artifact IDs, hashes, dispositions and dependency records |
| `manifest/` | Exhaustive release checksums and release metadata |

The submitted Supplementary Data 3 container is not duplicated here. Its archive and member-ledger identities are recorded, and all 32 identifier-free CSV extracts required by the supported workflows are included byte-exact with a public manifest.

## Repository map

```text
data/
  figure_sources/       frozen plotted-source tables and reader manifests
  processed/            identifier-free parsed model outputs released in SD3
  publication/          Source Data and Supplementary Data 1–2 workbooks
docs/                    figure, data, provenance and reproduction guidance
manifest/                exhaustive checksums and release metadata
provenance/              public-safe artifact IDs, hashes and dispositions
reference_outputs/main/ exact frozen Figure 1–5 PDFs
scripts/                 stable command-line entry points
src/ai4us/               reusable validation, analysis, and rendering code
tests/                    structural, numerical, and rendering checks
```

Figure 1 is author-supplied conceptual artwork and is not code-generated. See [the figure map](docs/figure_map.md) for all current main and supplementary figures.

## Reproduction scope

| Target | What is independently supported | Boundary |
|---|---|---|
| Source Data | Exact cell equality between 16 quantitative workbook sheets and their frozen CSV aliases | Workbook and CSV container bytes naturally differ |
| Figure 1 | Exact author-supplied conceptual PDF | Not code-generated |
| Figures 2–5 and S1–S12 | Scripted diagnostic rerenders from released plotted-source cells | Rerenders are not claimed to be pixel- or byte-identical to publication layouts |
| Primary Qwen2.5-VL-72B analysis | Released processed estimands, figure-source tables and exact reference output | Raw 72B inference, weights and hidden-state tensors are not part of the public workflow |
| Qwen2.5-VL-7B Perception replication | Standalone numerical recomputation from released identifier-free matrices | Starts after model inference and does not regenerate hidden states |

The supported workflows require only NumPy, pandas, Matplotlib and openpyxl. Raw API responses, raw model-inference artifacts, model weights, hidden-state tensors, images and restricted empirical record-level inputs are not redistributed. Third-party data remain governed by their original providers and terms. Compatibility ranges are in `pyproject.toml`; `requirements/locked.txt` records the exact clean-validation snapshot. Read the [data dictionary](docs/data_dictionary.md), [data provenance](docs/data_provenance.md), [model-rerun guidance](docs/model_reruns.md) and [reproduction status](docs/reproduction_status.md) before extending the analyses.

## Version and provenance

The manuscript reproducibility snapshot is pinned to commit [`6b710f8`](https://github.com/24kchengYe/AI4UrbanScience/commit/6b710f8e52ee0d68d759104610b842ff75913c03). Later documentation-only commits preserve that scientific snapshot in Git history. `manifest/files.sha256` and `manifest/release_manifest.json` define the public-tree identity; `scripts/verify_release.py` checks the release boundary, manifests and deterministic relationships.

## Citation, license and contact

Citation metadata for all six authors are provided in [`CITATION.cff`](CITATION.cff). Until the associated article receives a public DOI, please cite the repository together with the article title:

> Zhang, Y., Zhao, R., Huang, Z., Wang, X., Ma, Y. & Long, Y. *GenAI Models Capture Urban Science but Oversimplify Complexity*. AI4UrbanScience software release, version 2.0.0 (2026).

Code is released under the [MIT License](LICENSE). Data files and reference outputs may contain or derive from third-party sources and remain subject to the terms documented in the provenance files and associated article.

For scientific questions, open a GitHub issue. Correspondence about the study may be directed to Ying Long at `ylong@tsinghua.edu.cn`.
