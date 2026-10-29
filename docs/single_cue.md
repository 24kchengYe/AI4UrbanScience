# Recompute single-cue perception effects

This analysis compares tree-canopy, visible-pedestrian-activity and façade-repair edits with a same-scene editing control in 12 synthetic streetscapes. GPT-4o scored all 60 images on six 0–10 perception scales. The public files contain numerical scores and scene-level contrasts, not images.

From the repository root, with Python 3.10+ and NumPy:

```text
python -B scripts/recompute_single_cue.py
```

The command checks input hashes, reconstructs 60 prespecified scene-level contrasts, and prints five effects and their pointwise percentile 95% confidence intervals. It uses 10,000 scene resamples and seed 20260728, matching the stored summary and Supplementary Fig. 9d. It makes no network requests and writes no files.

| Edit → score | Mean effect | Pointwise 95% CI |
|---|---:|---:|
| Tree canopy → Beautiful | 1.208333 | [0.665625, 1.833333] |
| Tree canopy → Depressing | −1.525000 | [−2.358333, −0.666667] |
| Visible activity → Lively | 1.041667 | [0.541667, 1.541667] |
| Façade repair → Safe | 0.958333 | [0.750000, 1.166667] |
| Façade repair → Wealthy | 0.941667 | [0.675000, 1.250000] |

For each scene, the contrast is `(edited score − baseline score) − (editing-control score − baseline score)`, equivalent to edited minus editing-control score. Scenes, rather than individual images, are resampled. The five intervals are unadjusted for multiplicity. These are GPT-4o-predicted perception responses, not human-rated effects.

See [the data dictionary](../data/processed/single_cue/README.md) for all columns. The script retains the original effect definitions, bootstrap order and random seed; only file locations and input checks were adapted for this repository. Source identities are in [the provenance record](../provenance/recomputation_additions_20261029.json).
