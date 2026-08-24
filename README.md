# AI4UrbanScience

This repository accompanies **“GenAI Models Capture Urban Science but Oversimplify Complexity.”** It provides the released quantitative source tables, frozen reference figures, reader-oriented figure rerenders, and a standalone recomputation of the released 7B Perception state-replacement statistics.

The repository does not claim that every original model call or every publication-layout decision can be replayed from public assets. Exact submitted Figure 1–5 PDFs are retained under `reference_outputs/`; code-generated outputs are explicitly named `public_rerender`.

## Quick start

Python 3.10 or later is supported; Python 3.11 is used in continuous checks. The release verifier is a clean-checkout gate: it intentionally rejects environments, caches, build products, and rendered outputs inside the repository. Create the environment beside the clone and run without writing bytecode or a pytest cache.

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

The examples below use `python`; run them with the same external interpreter (or activate that external environment). Run the clean-checkout verification before creating any `outputs/` directory.

Render a smoke subset or the complete released-source set into a new output directory:

```bash
python scripts/reproduce_figures.py --figures 4,5,S10 --output-dir outputs/smoke
python scripts/reproduce_figures.py --all --output-dir outputs/all_figures
```

Recompute the released within-family 7B Perception state-replacement statistics:

```bash
python scripts/recompute_perception_state_replacement.py
```

Validate all 16 quantitative workbook sheets against their frozen CSV aliases:

```bash
python scripts/extract_source_data.py --report outputs/source_data_validation.json
```

## Repository map

```text
data/
  figure_sources/       frozen CSV inputs plus current reader manifests
  processed/            identifier-free parsed model outputs released in SD3
  publication/          Source Data and Supplementary Data 1–2 files
docs/                    figure map, provenance, scope, and rerun guidance
provenance/              public-safe artifact IDs, hashes, and dispositions
reference_outputs/main/ exact frozen Figure 1–5 PDFs
scripts/                 stable command-line entry points
src/ai4us/               reusable validation, analysis, and rendering code
tests/                    structural, numerical, and rendering checks
```

Figure 1 is author-supplied conceptual artwork and is not code-generated. See [the figure map](docs/figure_map.md) for all current main and supplementary figures.

## Reproduction scope

- Released workbook-to-CSV cell equality: verified for all 16 quantitative sheets.
- Figures 2–5 and S1–S12: public diagnostic rerenders from released figure-source tables; supplementary diagnostics render every released row with panel-specific labels.
- Figure 1: exact author-supplied conceptual artwork only.
- Perception 7B state replacement: numerical recomputation from released identifier-free matrices.
- Raw API/model inference, proprietary model outputs, model weights, hidden-state tensors, images, and restricted empirical record-level inputs: outside this public release.

The supported workflows require only NumPy, pandas, Matplotlib, and openpyxl. PyTorch and Transformers are not required because this release does not present raw model inference as a supported public workflow. Compatibility ranges are in `pyproject.toml`; `requirements/locked.txt` records the exact clean-validation snapshot. Figure content is reproducible from the released cells, but output-file byte identity and cross-version pixel identity are not claimed. Details are in [model reruns](docs/model_reruns.md) and [reproduction status](docs/reproduction_status.md).

## Data and citation

Read the [data dictionary](docs/data_dictionary.md) and [data provenance](docs/data_provenance.md) before extending the analyses. Citation metadata is provided in `CITATION.cff`. Code is released under the MIT License; third-party source data remain subject to their original terms.

The submitted Supplementary Data 3 archive is not bundled here. Its SHA-256 identity and member-ledger identity are recorded without publishing author-side locations; all 32 identifier-free CSV extracts needed by the supported workflows are bundled byte-exact with a public manifest.
