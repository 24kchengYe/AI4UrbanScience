# Reproduction status

| Target | Public workflow | Verified scope | Important boundary |
|---|---|---|---|
| Source Data | `scripts/extract_source_data.py` | Exact equality of all cells in 16 quantitative sheets; zero formulas | XLSX and CSV container hashes naturally differ |
| Figure 1 | Frozen PDF | Exact author-supplied conceptual artwork | Not code-generated |
| Figures 2–5 | `scripts/reproduce_figures.py` | Deterministic public rerender from released plotted-source cells | Not asserted to be pixel-identical to publication layouts |
| Figures S1–S12 | Same renderer | Every released source row is rendered; labels retain the panel-specific identifying fields | Some panels summarize heterogeneous metrics in a generic diagnostic layout |
| 7B Perception state replacement | `scripts/recompute_perception_state_replacement.py` | Correlation, bootstrap intervals, attenuation, specificity, sign-flip tests, and completeness checks | Starts from identifier-free released numerical matrices |
| Raw model generation | None | Not part of the supported public workflow | Requires external services or weights and non-released raw assets |

The exact Figure 1–5 PDFs are provided as visual references. A successful verification run demonstrates artifact integrity, workbook/source-table equality, public-boundary compliance, manifest consistency, tests, and rendering execution. It does not expand the release beyond the stated data boundary.

Current reader manifests live beside the released data. Author-side lineage is represented publicly only by stable artifact IDs and SHA-256 values; the complete location crosswalk and snapshots are outside this repository.

`requirements/locked.txt` records the exact clean-validation package set and Python version. `pyproject.toml` retains broader tested-compatible ranges for ordinary installation. Font, backend, and library changes can alter rendered bytes, so cross-version pixel identity is not asserted.
