# Experiment 05 — Synthetic Intervention via Controlled Image Generation

Use Nano Banana (an image generator/editor) to create baseline street-views
and three families of intervention variants (Natural / Traffic / Built).
The multimodal LLM then scores each image on six Place Pulse dimensions,
letting us quantify the causal effect of each intervention category.

## Pipeline

```
generate_images.py --mode baseline      → data/generated/perception_images/baseline/*.jpg
generate_images.py --mode variants      → data/generated/perception_images/variants/<cat>/*.jpg
clip_diversity.py                       → console: mean pairwise cosine distance
segment_and_analyze.py --step segment   → results/synthetic_intervention/segmentation_<mode>.csv
segment_and_analyze.py --step regression → results/synthetic_intervention/regression_<mode>.csv
segment_and_analyze.py --step delta     → results/synthetic_intervention/delta_scores.csv
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `17视觉感知实验-0纯图片生成（baseline）.py`       | `generate_images.py --mode baseline` |
| `17视觉感知实验-0纯图片生成（图改图 干预变体）.py` | `generate_images.py --mode variants` |
| `17色调标准化.py`                                | inline in `generate_images.py` (post-processing hook) |
| `17视觉感知实验-1CLIP验证多样性.py`               | `clip_diversity.py` |
| `19分割图片-0纯图片生成（baseline）.py`           | `segment_and_analyze.py --step segment --mode baseline` |
| `19分割图片-0纯图片生成（图改图 干预变体）.py`      | `segment_and_analyze.py --step segment --mode variants` |
| `20统计分析-0纯图片生成（baseline）.py`           | `segment_and_analyze.py --step regression --mode baseline` |
| `20统计分析-0纯图片生成（图改图 干预变体）.py`      | `segment_and_analyze.py --step delta` |

## Prompt variants

Baseline: `image.baseline_single` (minimal) or `image.structured_diverse`
(the 6-lever structured prompt used in the paper).

Intervention prompts for the three element families are registered as
`image.intervention_natural`, `image.intervention_traffic`,
`image.intervention_built`.

## Expert-evaluation Flask tool

The optional expert-scoring UI from ``Expert_Evaluation_Tool/evaluate_app.py``
is retained in ``expert_tool/`` for reference; run it locally with
``python experiments/05_synthetic_intervention/expert_tool/evaluate_app.py``.
