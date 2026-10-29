# A controlled virtual urban experiment

The [Scaling example](../examples/controlled_scaling/) asks how GDP and Road respond when Population changes within a fictional city. Published metropolitan relationships motivate the variables and comparison. The intervention requests GDP in million generic currency units and total road length in kilometres; these are distinct from the lane-mile and metropolitan-product definitions in the separate synthesis experiment.

## Design and run

1. Hold region, climate, topography, economic base and planning history fixed within each city setting.
2. Request GDP and Road at Population levels of 50,000, 100,000, 200,000, 400,000 and 800,000 people. The exact template is in `examples/controlled_scaling/prompt_template.json`; its public prompt key is `SCL-INT-PRIMARY_SCHEMA` in Supplementary Data 1.
3. Fit log10(GDP) and log10(Road) against log10(Population) within each setting. Compare the two slopes, then summarize over 100 settings.

From the repository root:

```text
python -B examples/controlled_scaling/run_example.py --output-dir ../ai4us-scaling-results
```

Use Python 3.10+ and a fresh output directory outside the checkout. No additional packages are needed. The script analyses the 500 existing primary responses and checks their means against the Figure 4 source table:

| Result | Recomputed value |
|---|---:|
| Mean Road exponent | 0.6815397449294656 |
| Mean GDP exponent | 1.0338220034025647 |
| Settings with GDP exponent greater than Road | 99 / 100 |

`setting_estimates.csv` contains the setting-level results; `summary.json` contains the means and checks. The original 2,100-row CSV also retains alternative wording, no-change and irrelevant-number conditions. This entry point recomputes primary point estimates; their uncertainty and control comparisons are in Source Data and Methods 4.6.

## Adapt the design

Choose an empirical reference, variables, range of change, fixed context, control conditions, response measure and missing-output rule before collecting new outputs. The same design uses distance bands for Decay, Functional Mix for Vitality, and matched streetscape edits for Perception. Responses can suggest hypotheses for empirical investigation; they do not by themselves estimate real-world causal effects.

The example supplies the prompt structure and analysis, not an inference client. New model requests need urban settings and model-access configuration. For a second offline analysis, see [single-cue perception effects](single_cue.md).

## Prompt source

The original `prompt_template.json` is retained byte-for-byte, including its source-workbook name. Its public repository equivalent is [Supplementary Data 1](../data/publication/20261029/supplementary_data_1_prompts_and_inference.xlsx), sheet **Prompt templates**, Excel row **13** (including the header row), key **SCL-INT-PRIMARY_SCHEMA**. All eleven record fields match that row. The workbook SHA-256 is `93c0070bfd4b1f8e27ab5276e4d1b240c65f4163f436965b574df0050d1dcd9c`; the unchanged template SHA-256 is `105251e85dcf3583fa9ddb0553e518b8f656527cc69a91df33f102632d8b04cb`.
