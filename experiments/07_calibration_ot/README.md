# Experiment 07 — LLM + Optimal Transport Calibration

Ablation study showing how distribution-level calibration via 1-D and joint
Optimal Transport bends GenAI-produced scaling exponents back toward the
Bettencourt-2013 benchmark.

## Pipeline

```
calibration.py → results/calibration_ot/ablation.csv
```

## Prerequisites

Requires the real-world city CSV produced by
`experiments/06_distribution_divergence/extract_real_data.py`, plus a set of
generated replicates from Experiment 01 (`generate.py`).

## Methods compared

| method | description |
|---|---|
| `raw`            | LLM output without any calibration |
| `quantile_match` | per-column quantile alignment |
| `ot_marginal`    | 1-D Sinkhorn Optimal Transport, one column at a time |
| `ot_joint`       | multidimensional Sinkhorn OT over all columns jointly |

## Quick start

```bash
python experiments/07_calibration_ot/calibration.py
```

## Legacy script mapping

| Legacy script | New entry point |
|---|---|
| `13生成数据分布矫正.py`          | `calibration.py` (`--methods quantile_match`) |
| `14消融实验 llm+OT.py`           | `calibration.py --methods raw ot_marginal ot_joint` |
| `15标度系数验证 llm+OT.py`       | `calibration.py` (prints Bettencourt comparison in output table) |
| `16不同方法的数据规律拟合可视化.py`| consumed inside `calibration.py` as tabular output |
| `16真实数据校准SCforAI4US.py`    | `calibration.py` (real-world anchor via `--real-csv`) |
