# Experiment 04 — Human-AI Perception Alignment (Fig. 3a, 3b)

Benchmark a multimodal LLM against the MIT Place Pulse 2.0 crowdsourced
dataset. For each of six perceptual dimensions we sample 1000 random image
pairs, ask the AI to pick LEFT / RIGHT / EQUAL, then compare to human votes
with Cohen's Kappa.

## Pipeline

```
pairwise_eval.py     → data/generated/perception_alignment/<model>/<dim>.csv
kappa_scores.py      → results/perception_alignment/<model>_kappa.csv (console summary)
trueskill_scoring.py → results/perception_alignment/<model>/trueskill/<dim>.csv
figure3.py           → results/figures/figure3.png + .pdf
```

## Data prerequisite

Download the Place Pulse 2.0 dataset and set `PLACE_PULSE_DIR` in `.env`:

```
PLACE_PULSE_DIR=/absolute/path/to/place_pulse_v2
```

The directory must contain `votes.csv` and `images/<image_id>.jpg`.

## Quick start

```bash
python experiments/04_perception_alignment/pairwise_eval.py
python experiments/04_perception_alignment/kappa_scores.py
python experiments/04_perception_alignment/figure3.py
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `18感知评分-验证AI"感知能力" (pulse随机抽样六维度感知).py` | `pairwise_eval.py` |
| `18感知评分-验证AI"感知能力"..可视化-kappa系数打分.py`     | `kappa_scores.py` + `figure3.py` |
| `18感知评分和trueskill.py`                             | `trueskill_scoring.py` |
| `18感知评分和trueskill-Q-score.py`                     | `trueskill_scoring.py` (the Q-score variant is collapsed into one script) |
| `20统计分析-感知部分-naturecities改稿用-图3.py`          | `figure3.py` |
