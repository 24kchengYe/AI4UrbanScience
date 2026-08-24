# Data dictionary

All CSV files are UTF-8, comma-delimited tables with one header row. Empty cells represent fields that do not apply to that record or an unavailable interval; they should not be silently converted to zero.

## Figure-source tables

`data/figure_sources/` contains one long-form quantitative table per current figure. The same table is available as a named sheet in `data/publication/source_data.xlsx`.

| Field family | Meaning |
|---|---|
| `panel` | Figure panel identifier; panel-specific record types may share a table |
| `benchmark` or `theory` | Urban-science case represented by the row |
| `condition`, `source`, `series` | Compared model, empirical, prompt, or intervention condition |
| `metric`, `parameter`, `feature`, `dimension` | Quantity summarized by the row |
| `estimate` | Plotted point estimate |
| `ci_low`, `ci_high` | Lower and upper interval endpoints where available |
| `closed_value`, `open_value` | Paired closed- and open-model values used in Figure 5/S10 |
| `decoder_block`, `selected` | Open-model readout location and prespecified selected block |

Panel-specific columns are documented by their table headers and preserved exactly from the frozen Source Data cells. `scripts/extract_source_data.py` verifies the workbook/CSV mapping.

Each figure-source directory also has a `manifest.json` containing only current public paths plus a content-addressed identity for author-side lineage.

## Parsed model outputs

`data/processed/model_outputs/` is the extracted, identifier-free content of Supplementary Data 3.

| File group | Unit represented | Key fields |
|---|---|---|
| `urban_scaling_*generation_attempts.csv` | One requested generated table | model/condition, parse status, requested size, summary counts |
| `urban_scaling_*generation_rows.csv` | One parsed city row | released city cell, population, road/GDP outcomes |
| `urban_scaling_intervention_outputs.csv` | One controlled input/output request | condition, carrier, changed input, parsed response |
| `urban_scaling_short_answer_activation_estimands.csv` | One activation-edit estimand | target relation, contrast, estimate and interval |
| `distance_decay_generation_attempts.csv` | One requested radial profile | condition, parse/fit status and summary metrics |
| `distance_decay_generation_profiles.csv` | One released radial-profile cell | generated profile, annulus distance and density ratio |
| `distance_decay_generation_parameters.csv` | One fitted generated profile | fitted radial parameters and diagnostics |
| `distance_decay_intervention_outputs.csv` | One distance-change output cell | released grid/carrier, distance band, parsed ratio |
| `vitality_generation_attempts.csv` | One requested neighbourhood table | condition, parse status and row counts |
| `vitality_generation_rows.csv` | One parsed neighbourhood row | released predictors and generated activity outcome |
| `vitality_intervention_requests.csv` | One controlled vitality request | carrier, intervention factor and requested direction |
| `vitality_intervention_pair_results.csv` | One paired vitality result | control/target outputs and directional response |
| `vitality_short_answer_activation_estimands.csv` | One vitality activation estimand | contrast, estimate and interval |
| `open_closed_model_profile_means.csv` | One matched behavioural cell | theory, cell and closed/open means |
| `complete_table_activation_effects.csv` | One complete-table activation contrast | theory, stage, effect and validation fields |
| `perception_generation_representation_*.csv` | One Perception representation summary/sensitivity row | encoder/block, metric, estimate and interval |
| `perception_human_alignment_*.csv` | One task or dimension-level alignment row | dimension, human reference and model metric |
| `perception_intervention_*.csv` | One intervention score, scene estimand, or QC row | scene, theme, dimension, condition and effect |
| `perception_activation_*.csv` | One activation-effect cell or scene estimand | scene, theme, dimension, condition and effect |
| `perception_7b_behavior_*.csv` | One released behaviour cell or summary | scene, theme, dimension, reference and signed score |
| `perception_7b_state_replacement_*.csv` | One state-replacement cell, scene estimand, or summary | scene, target/random patch, equal-norm check and effect |
| `perception_7b_replication_summary.csv` | Registered/usable unit counts | analysis component and completeness counts |

The exact headers are authoritative. Public extract hashes are recorded in `data/release_manifests/parsed_model_outputs_manifest.csv`; the external archive identity is in `data/release_manifests/parsed_model_outputs_reference.json`; and public-safe lineage is in `provenance/artifact_lineage.csv`.
