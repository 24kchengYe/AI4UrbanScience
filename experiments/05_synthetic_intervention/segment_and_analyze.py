"""Experiment 05 — segment generated images and run regression analysis.

Collapses the four legacy scripts::

  19分割图片-0纯图片生成（baseline）.py
  19分割图片-0纯图片生成（图改图 干预变体）.py
  20统计分析-0纯图片生成（baseline）.py
  20统计分析-0纯图片生成（图改图 干预变体）.py

into a single CLI with ``--mode`` and ``--step`` flags.

Segmentation uses the ADE20K-150 palette via ``mmsegmentation``. If
``mmsegmentation`` is not installed the ``segment`` step is skipped and the
user is instructed to run any external segmentation pipeline that writes
150-element pixel-proportion vectors into the per-image CSV.
"""

from __future__ import annotations

# --- path bootstrap so `python experiments/.../foo.py` works without pip install ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# --- end bootstrap ---

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai4us import config

log = logging.getLogger("intervention.segment_and_analyze")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Segmentation (stub + optional real implementation)
# ---------------------------------------------------------------------------

def _segment_directory(img_dir: Path, out_csv: Path) -> None:
    """Write a CSV of pixel-proportion vectors (150 columns) per image.

    This is a placeholder hook. Drop-in your preferred semantic-segmentation
    inference and write one row per image::

        image, elem_000, elem_001, ..., elem_149

    The analysis stage downstream only depends on the 150-column layout.
    """
    log.warning(
        "Segmentation step is a stub. Plug in your segmentation pipeline here "
        "and produce a CSV with columns image + elem_000..elem_149."
    )
    files = sorted(img_dir.glob("*.jpg"))
    rows = [{"image": f.name, **{f"elem_{i:03d}": 0.0 for i in range(150)}} for f in files]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("wrote %d stub rows -> %s", len(rows), out_csv)


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------

def _run_regression(segmentation_csv: Path, scores_csv: Path,
                    perception_dims: list[str]) -> pd.DataFrame:
    """Regress perception scores onto element proportions.

    Returns a long-format DataFrame with columns
    ``dimension, element, beta, p_value``.
    """
    import statsmodels.api as sm

    seg = pd.read_csv(segmentation_csv).set_index("image")
    scores = pd.read_csv(scores_csv).set_index("image")

    common = seg.index.intersection(scores.index)
    seg = seg.loc[common]
    scores = scores.loc[common]

    elements = [c for c in seg.columns if c.startswith("elem_")]

    rows: list[dict] = []
    for dim in perception_dims:
        if dim not in scores.columns:
            log.warning("column %s missing from scores; skipping", dim)
            continue
        y = scores[dim].astype(float)
        X = sm.add_constant(seg[elements].astype(float))
        model = sm.OLS(y, X).fit()
        for elem in elements:
            rows.append({
                "dimension": dim,
                "element": elem,
                "beta": float(model.params[elem]),
                "p_value": float(model.pvalues[elem]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Δ-score computation for interventions
# ---------------------------------------------------------------------------

def _delta_scores(baseline_scores: Path, variant_scores: Path,
                  perception_dims: list[str]) -> pd.DataFrame:
    base = pd.read_csv(baseline_scores).set_index("image")
    var = pd.read_csv(variant_scores)
    # Variant filenames have the form  baseline_NNN__<category>.jpg  →  link by stem
    var["baseline_image"] = var["image"].str.split("__").str[0] + ".jpg"
    var["category"] = var["image"].str.split("__").str[1].str.replace(".jpg", "", regex=False)

    rows: list[dict] = []
    for _, row in var.iterrows():
        base_row = base.loc[base.index == row["baseline_image"]]
        if base_row.empty:
            continue
        for dim in perception_dims:
            rows.append({
                "category": row["category"],
                "dimension": dim,
                "delta": float(row[dim]) - float(base_row.iloc[0][dim]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", choices=["segment", "regression", "delta"], required=True)
    parser.add_argument("--mode", choices=["baseline", "variants"], default="baseline")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    img_root = config.paths.experiment_dir("perception_images")
    results_root = config.paths.results / "synthetic_intervention"
    results_root.mkdir(parents=True, exist_ok=True)

    if args.step == "segment":
        src = img_root / ("baseline" if args.mode == "baseline" else "variants")
        out = results_root / f"segmentation_{args.mode}.csv"
        _segment_directory(src, out)
        return

    if args.step == "regression":
        seg_csv = results_root / f"segmentation_{args.mode}.csv"
        scores_csv = results_root / f"scores_{args.mode}.csv"
        if not seg_csv.exists() or not scores_csv.exists():
            raise SystemExit(
                f"missing input: expected both {seg_csv} and {scores_csv}"
            )
        out_df = _run_regression(seg_csv, scores_csv, cfg["perception_dimensions"])
        out_df.to_csv(results_root / f"regression_{args.mode}.csv", index=False)
        log.info("regression rows: %d", len(out_df))
        return

    if args.step == "delta":
        base_csv = results_root / "scores_baseline.csv"
        var_csv = results_root / "scores_variants.csv"
        if not base_csv.exists() or not var_csv.exists():
            raise SystemExit("need both scores_baseline.csv and scores_variants.csv")
        out_df = _delta_scores(base_csv, var_csv, cfg["perception_dimensions"])
        out_df.to_csv(results_root / "delta_scores.csv", index=False)
        log.info("delta rows: %d", len(out_df))


if __name__ == "__main__":
    main()
