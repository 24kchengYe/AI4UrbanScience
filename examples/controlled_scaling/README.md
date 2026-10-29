# Controlled Scaling example

This example computes how Road and GDP change as Population varies within 100 fictional-city settings. It contains the original prompt template, design contract and numerical responses.

From the repository root, using Python 3.10 or later:

```text
python -B examples/controlled_scaling/run_example.py --output-dir ../ai4us-scaling-results
```

Choose a fresh directory outside the checkout. The example uses the standard library, runs offline and writes `setting_estimates.csv` and `summary.json`. It selects 500 primary responses from the 2,100 archived records. The mean Road exponent is 0.6815397449294656 and the mean GDP exponent is 1.0338220034025647; GDP exceeds Road in 99 of 100 settings.

See [the worked-example guide](../../docs/worked_example.md) for the design, units, controls and instructions for adapting the question. `input_manifest.json` verifies the unchanged data, contract and prompt before computation. The CSV data and original prompt text are byte-identical to the accompanying example archive; [provenance](../../provenance/recomputation_additions_20261029.json) records their source identities.
