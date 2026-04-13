# Experiment 02 — Urban Distance Decay (meso-scale)

Generate concentric-ring density profiles for a city and test whether the
resulting surface fits the inverse-S density model (Clark 1951, Bussière 1972,
et al.).

## Pipeline

```
generate.py   → data/generated/distance_decay/<model>/<prompt>/run_NNN.xlsx
analyze.py    → results/distance_decay/<model>/<prompt>.csv
visualize.py  → results/figures/distance_decay.png + .pdf
```

## Quick start

```bash
python experiments/02_distance_decay/generate.py
python experiments/02_distance_decay/analyze.py
python experiments/02_distance_decay/visualize.py
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `07test-openai(distance).py`                              | `generate.py --model gpt-4o` |
| `07test-claude(distance) .py`                             | `generate.py --model claude-3.5-sonnet` |
| `07test-deepseek(distance) .py`                           | `generate.py --model deepseek-v3` |
| `07test-chatglm(distance) .py`                            | `generate.py --model chatglm-4` |
| `07test-gemini-*(distance).py`                            | `generate.py --model gemini-2.0-flash` |
| `08AIUS-TheoryValid(distance).py`                         | `analyze.py` (single replicate) |
| `09AIUS-TheoryValidMultitime(distance).py`                | `analyze.py` (aggregated) |
| `09AIUS-visualization between theory and G(distance).py`  | `visualize.py` |
| `09AIUS-visualization ... 2八环合一.py`                    | `visualize.py` (8-ring concatenation variant) |
| `09AIUS-visualization ... Linear statistics.py`           | linear-statistics variant, superseded by `analyze.py` |
| `09AIUS-visualization of G from Jiaolimin(distance).py`   | see `visualize.py` (custom marker data) |

All six prompt variants (baseline 100-ring, 150-ring detailed, real-world
anchor, theory anchor, real+theory combined) are registered under
`distance_decay.*` in `ai4us/prompts.py`.
