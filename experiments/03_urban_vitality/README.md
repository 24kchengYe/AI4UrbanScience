# Experiment 03 — Urban Vitality (Jacobs, micro-scale)

Two-stage experiment: the model first generates block attributes, then
takes its own output back as input and assigns a livability score to each
block. We test whether the OLS regression of livability on the five Jacobs
attributes has the signs predicted by Jacobs' theory.

## Pipeline

```
generate_attributes.py   → data/generated/vitality_attributes/<model>/<prompt>/run_NNN.xlsx
score_livability.py      → data/generated/vitality_scored/<model>/<scoring_prompt>/run_NNN.xlsx
analyze.py               → results/vitality/<model>/<scoring_prompt>.csv
figure2.py               → results/figures/figure2.png + .pdf
```

## Quick start

```bash
# Step 1: generate blocks (writes 100 replicates of 100 blocks each)
python experiments/03_urban_vitality/generate_attributes.py

# Step 2: have the model score each block's livability
python experiments/03_urban_vitality/score_livability.py

# Step 3: fit OLS across replicates and print a summary
python experiments/03_urban_vitality/analyze.py

# Step 4: produce Fig. 2
python experiments/03_urban_vitality/figure2.py
```

## Prompt variants

Step 1 (attribute generation) has two variants in the registry
(`vitality_attributes.jacobs_five` — the paper's primary five-element
schema — and `vitality_attributes.street_density_scale_age` — the early
alternate schema with street density and block scale).

Step 2 (scoring) has five variants mixing persona (`you are a resident`),
theory anchor (`refer to Jane Jacobs' theory`) and output format (JSON vs
weighted list):

- `vitality_scoring.direct_json`
- `vitality_scoring.as_resident`
- `vitality_scoring.jacobs_theory`
- `vitality_scoring.resident_plus_theory`
- `vitality_scoring.resident_with_weights`

Select any of them on the command line:

```bash
python experiments/03_urban_vitality/score_livability.py \
    --scoring-prompt vitality_scoring.jacobs_theory
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `04test-openai(vitality).py`                        | `generate_attributes.py --model gpt-4o` |
| `04test-claude(vitality).py`                        | `generate_attributes.py --model claude-3.5-sonnet` |
| `04test-deepseek(vitality).py`                      | `generate_attributes.py --model deepseek-v3` |
| `04test-gemini-*(vitality).py`                      | `generate_attributes.py --model gemini-2.0-flash` |
| `04test-chatglm(vitality).py`                       | `generate_attributes.py --model chatglm-4` |
| `04test-*(vitality access).py`                      | `score_livability.py --model <model>` |
| `05AIUS-TheoryValid(vitality).py`                   | `analyze.py` |
| `06AIUS-TheoryValidMultitime(vitality).py`          | `analyze.py` + `figure2.py` |
| `06AIUS-TheoryValidMultitime(vitality)虚线和黑长直*`| superseded by `figure2.py` |
| `06AIUS-TheoryValidMultitime...naturecities改稿用-图2` | superseded by `figure2.py` |
