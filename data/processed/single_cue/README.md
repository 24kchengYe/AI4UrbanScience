# Single-cue numerical data

The files preserve the accompanying Supplementary Data 3 numerical records byte-for-byte. Twelve pseudonymous scene IDs (`singlecue_scene_001` through `singlecue_scene_012`) link conditions; no original image names, locations or source images are included.

| File | Rows | Contents |
|---|---:|---|
| `perception_single_cue_scores.csv` | 60 | One row per scene and image condition; six GPT-4o scores on 0–10 scales |
| `perception_single_cue_scene_estimands.csv` | 60 | Twelve scenes × five prespecified cue–score contrasts |
| `perception_single_cue_summary.csv` | 5 | Mean effects and scene-bootstrap intervals |
| `perception_single_cue_qc.csv` | 48 | Archived identity-preservation and manipulation checks for the four edited/control conditions per scene |
| `single_cue_analysis_config.json` | — | Ordered cue–score links, scene IDs, bootstrap seed and resample count |
| `figure_s9_source.csv` | — | Unchanged source table for Supplementary Fig. 9; panel `d` provides the five comparison estimates |

## Column definitions

- `scene_id` is the same pseudonymous scene key across files. `condition` is `baseline`, `sham` (same-scene editing control), `tree_canopy`, `visible_activity` or `facade_repair`.
- `Beautiful`, `Depressing`, `Lively`, `Safe`, `Wealthy` and `Boring` are the six predicted perception scores, not human measurements.
- `dimension` identifies the score being compared; `direction` is its prespecified `up` or `down` expectation.
- `target_change` is edited minus baseline score; `sham_change` is editing-control minus baseline score; `estimand` is their difference.
- `estimate` is the mean of 12 scene effects; `ci95_low` and `ci95_high` are pointwise, unadjusted percentile 95% bootstrap bounds. `n_scenes` is the denominator; `direction_supported_by_point_estimate` compares the mean sign with the prespecified direction.
- `identity_preserved`, `identity_confidence`, `manipulation_pass` and `manipulation_confidence` retain the archived check outputs. They are not used to change the 12-scene analysis denominator.
- In the figure source table, `panel`, `condition` and `dimension` identify a plotted result; `estimate`, `ci_low` and `ci_high` hold its mean and interval. The recomputation reads panel `d` only.

`input_manifest.json` verifies the six input files. Run `python -B scripts/recompute_single_cue.py` from the repository root; see [the analysis guide](../../../docs/single_cue.md).
