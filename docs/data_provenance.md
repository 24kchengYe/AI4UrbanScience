# Data provenance and public boundary

The public tree is built from the frozen Source Data workbook, Supplementary Data 1–2, frozen figure-source tables, and identifier-free parsed model outputs. `provenance/artifact_lineage.csv` records the original 59 public-safe artifact identities. The original 131-row disposition is in `provenance/dependency_inventory.csv`. Both tables use content-addressed artifact IDs rather than author-side locations. Version 2.1.0 additions are recorded separately so these historical inventories remain unchanged.

## Public artifacts

- `data/publication/source_data.xlsx` contains the 16 quantitative sheets used by Figures 2–5 and S1–S12.
- `data/figure_sources/` contains cell-matrix-equivalent CSV aliases and current reader manifests that point only to paths present in this repository.
- `data/processed/model_outputs/` contains 32 identifier-free CSV extracts for direct analysis.
- `data/release_manifests/parsed_model_outputs_manifest.csv` records their public paths, sizes, row counts, and hashes.
- `data/release_manifests/parsed_model_outputs_reference.json` records the submitted Supplementary Data 3 archive and member-ledger identities by SHA-256. Those two external containers are not bundled.
- `data/publication/20261029/` holds byte-exact copies of the revised prompt/inference and empirical-provenance workbooks. `provenance/workbook_update_20261029.json` records their hashes and cell-level differences from the original workbooks.
- `examples/controlled_scaling/` and `data/processed/single_cue/` contain the additional offline-recomputation inputs. Their source archive identities, member hashes and adaptation scope are in `provenance/recomputation_additions_20261029.json`.
- `provenance/supplementary_label_update_20261029.json` records the supplementary-label changes and reference PDFs.

The source-workbook adapter verifies all headers and cells, not byte identity between XLSX and CSV containers. The frozen validation result is 16 of 16 equal sheets with zero formula cells.

## Artifact lineage

Current reader manifests retain content-addressed identities for their author-side lineage records without exposing author-side locations. The complete location crosswalk and byte-exact snapshots are retained outside the public repository.

The manifests beside the source tables identify the current `ai4us.figures` module, stable command, released source table, and—where bundled—the exact main-figure reference PDF. They explicitly do not claim that the diagnostic renderer reconstructs the publication layout byte-for-byte or pixel-for-pixel.

## Inputs not redistributed

Five exact historical tables are intentionally absent:

| Withheld exact table | Reason | Public replacement |
|---|---|---|
| City-level real/generated profile metrics | Contains city-level values derived from empirical GISA profiles | Aggregate `Fig03_data` cells |
| Distance-decay source curves | Mixes generated and empirical GISA profiles | `Fig02_data` plus SD3 `distance_decay_generation_profiles.csv` |
| Distance-decay source parameters | Mixes generated and published empirical parameter rows | `Fig02_data` plus SD3 `distance_decay_generation_parameters.csv` |
| 52-site vitality predictor records | City of Melbourne record-level data are not redistributed | Aggregate `Fig03_data` cells |
| 52-site pedestrian-activity records | City of Melbourne record-level data are not redistributed | Aggregate `Fig03_data` cells |

These exclusions preserve the declared empirical-data boundary. They also mean that publication-layout Figures 2 and 3 cannot be rebuilt from raw unit-level records using only this repository. Their released aggregate cells can be validated and rerendered.

## Third-party sources

The revised [Supplementary Data 2](../data/publication/20261029/supplementary_data_2_empirical_provenance.xlsx) contains source-specific provenance, retrieval, license, and redistribution notes for empirical references. Its empirical records are unchanged from the original workbook. The MIT License applies to repository code, not to third-party datasets.
