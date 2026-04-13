# Experiment 01 — Urban Scaling Law (Fig. 1)

Test whether GenAI-generated urban data reproduces the classical power-law
relations of **urban scaling laws** (Bettencourt 2013) across 100 cities and
100 independent replicates.

This experiment maps to Fig. 1 of the manuscript.

## Pipeline

```
generate.py         →   data/generated/scaling_law/<model>/<prompt>/run_NNN.xlsx
analyze.py          →   results/scaling_law/<model>/<prompt>.csv
sensitivity.py      →   data/generated/scaling_law/<model>/sensitivity/<mode>_bs<N>/
figure1.py          →   results/figures/figure1.png + figure1.pdf
```

## Quick start

Generate 100 replicates with the primary model (GPT-4o) and the baseline prompt:

```bash
python experiments/01_urban_scaling_law/generate.py
```

Fit Zipf / infrastructure / GDP scaling on those replicates and print a
summary table against the Bettencourt-2013 benchmark:

```bash
python experiments/01_urban_scaling_law/analyze.py
```

Produce Fig. 1:

```bash
python experiments/01_urban_scaling_law/figure1.py
```

## Running the full model sweep

```bash
python experiments/01_urban_scaling_law/generate.py --all-models
```

## Sensitivity analysis (independent vs joint sampling)

```bash
python experiments/01_urban_scaling_law/sensitivity.py --mode joint --batch-size 100
python experiments/01_urban_scaling_law/sensitivity.py --mode independent --batch-size 100
```

## Configuration

All defaults live in `config.yaml`. Override anything on the command line:

```bash
python experiments/01_urban_scaling_law/generate.py \
    --model claude-3.5-sonnet \
    --prompt scaling_law.with_theory \
    --n-replicates 50
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `01test-openai(Zipf-Scaling).py`                   | `generate.py --model gpt-4o` |
| `01test-openai-智匠AI(Zipf-Scaling).py`            | `generate.py --model gpt-4o-mindcraft` |
| `01test-claude(Zipf-Scaling).py`                   | `generate.py --model claude-3.5-sonnet` |
| `01test-deepseek(Zipf-Scaling).py`                 | `generate.py --model deepseek-v3` |
| `01test-chatglm(Zipf-Scaling).py`                  | `generate.py --model chatglm-4` |
| `01test-gemini-*.py`                               | `generate.py --model gemini-2.0-flash` |
| `01test-qwen-*.py`                                 | `generate.py --model qwen-plus` |
| `01test-Doubao-*.py`                               | `generate.py --model doubao-pro` |
| `01test-hunyuan-*.py`                              | `generate.py --model hunyuan-large` |
| `01test-o1-preview-*.py`                           | `generate.py --model o1-preview` |
| `01test-sensitivity-独立采样*.py`                   | `sensitivity.py --mode independent` |
| `01test-sensitivity-联合采样*.py`                   | `sensitivity.py --mode joint` |
| `02AIUS-TheoryValid(Zipf-Scaling).py`              | `analyze.py` (single-replicate mode) |
| `03AIUS-TheoryValidMultitime(Zipf-Scaling).py`     | `analyze.py` (full aggregation) |
| `03AIUS-TheoryValidMultitime趋势线和置信区间*.py`    | superseded by `figure1.py` |
| `03AIUS-TheoryValidMultitime-naturecities改稿用-图1.py` | superseded by `figure1.py` |

Eleven prompt variants (p1, p2 China 2020, p3 real-world reference, p4 theory
guidance, …) are all preserved in `ai4us/prompts.py` under the `scaling_law.*`
namespace. See that file for the full list.
