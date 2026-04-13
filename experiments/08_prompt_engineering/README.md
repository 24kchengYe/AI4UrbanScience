# Experiment 08 — SynAlign Prompt Engineering

Four-stage blueprint-prompting pipeline that tests whether injecting a
"statistical blueprint" (symbolic tasks) or a "perceptual mapping
blueprint" (image tasks) improves GenAI fidelity on urban data.

## Pipelines

```
pipeline_symbolic.py   → data/generated/prompt_engineering_symbolic/{with,without}_blueprint/
                      → results/prompt_engineering_symbolic/ablation.csv
pipeline_perceptual.py → data/generated/prompt_engineering_perceptual/<dim>_with_blueprint.csv
                      → results/prompt_engineering_perceptual/stage4_evaluation.csv
```

Each pipeline runs four stages:

1. **Discovery**   — the model writes a blueprint summarising real-world
   regularities.
2. **Synthesis**   — the model re-generates data with the blueprint in-context.
3. **Alignment**   — we fit/evaluate the generated data against theoretical
   benchmarks or human ground truth.
4. **Evaluation**  — pass if a threshold fraction of the evaluation passes.

## Quick start

```bash
python experiments/08_prompt_engineering/pipeline_symbolic.py
python experiments/08_prompt_engineering/pipeline_perceptual.py
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `16参考SynAlign01-urban_stage1_discovery.py`              | `pipeline_symbolic.stage1_discovery()` |
| `16参考SynAlign02-urban_stage2_synthesis.py`              | `pipeline_symbolic.stage2_synthesis()` |
| `16参考SynAlign03-urban_stage3_alignment.py`              | `pipeline_symbolic.stage3_alignment()` |
| `16参考SynAlign04-urban_stage4_evaluation.py`             | `pipeline_symbolic.stage4_evaluation()` |
| `16参考SynAlign05-urban_stage1_discovery.py` (perceptual) | `pipeline_perceptual.stage1_blueprint()` |
| `16参考SynAlign06-urban_stage2_synthesis.py`              | `pipeline_perceptual.stage2_pairwise()` |
| `16参考SynAlign07-urban_stage3_alignment.py`              | `pipeline_perceptual.stage3_kappa()` |
| `16参考SynAlign08-urban_stage4_evaluation.py`             | final block of `pipeline_perceptual.main()` |
| `SynAlign06…without synalign.py`                          | `pipeline_symbolic.py` (automatic baseline arm) |
| `SynAlign06…人类知识注入提示增强版本.py`                    | `pipeline_symbolic.py` with `--use-blueprint` (default) |
| `SynAlign00-synthesis…symbolic.py`                        | `pipeline_symbolic.stage2_synthesis()` |
| `SynAlign00-synthesis…symbolic-并行1/2/3.py`               | `pipeline_symbolic.py` (single-pass; parallelism is a deployment concern not a code concern) |
| `SynAlign00-synthesis…perceptual.py`                      | `pipeline_perceptual.py` |
| `SynAlign00-synthesis…可视化*`                             | downstream of `pipeline_*`; plug into `figure_si.py` (not yet written — out of scope for Fig. 1-4) |
