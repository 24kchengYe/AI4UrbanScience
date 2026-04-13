"""Experiment 04 — produce Fig. 3a/3b of the manuscript.

Two panels:
 a) Confusion matrices (one per dimension) + bar chart of Cohen's Kappa.
 b) Bar chart of standardized regression coefficients from
    ``multiple_regression.py`` (Fig. 3b of the manuscript).

Fig. 3c (the synthetic-intervention panel) is produced by
``experiments/05_synthetic_intervention/figure3c.py``.
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ai4us import config, viz

CONFIG_PATH = Path(__file__).parent / "config.yaml"
LABELS = ["LEFT", "RIGHT", "EQUAL"]


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    mapping = {"LEFT": 0, "RIGHT": 1, "EQUAL": 2}
    for _, row in df.iterrows():
        h = mapping.get(str(row["human_choice"]).upper())
        a = mapping.get(str(row["ai_choice"]).upper())
        if h is not None and a is not None:
            cm[h, a] += 1
    return cm


def _cohens_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = ((cm.sum(axis=0) / n) * (cm.sum(axis=1) / n)).sum()
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--output", type=Path,
                        default=config.paths.figures / "figure3.png")
    args = parser.parse_args(argv)

    viz.use_paper_style(font_size=11)
    in_dir = config.paths.experiment_dir("perception_alignment") / args.model
    if not in_dir.exists():
        raise SystemExit(f"no judgments at {in_dir}; run pairwise_eval.py first")

    dims = cfg["dimensions"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    kappas: list[float] = []
    for i, dim in enumerate(dims):
        ax = axes[i // 3, i % 3]
        f = in_dir / f"{dim}.csv"
        if not f.exists():
            ax.set_title(f"{dim}\n(missing)")
            kappas.append(float("nan"))
            continue
        df = pd.read_csv(f)
        cm = _confusion_matrix(df)
        k = _cohens_kappa(cm)
        kappas.append(k)
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() * 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        ax.set_xlabel("AI"); ax.set_ylabel("Human")
        ax.set_title(f"{dim}  (κ={k:.2f})")

    # Last cell = Kappa bar chart across dimensions
    ax = axes[1, 3]
    ax.bar(range(len(dims)), kappas,
           color=[viz.PALETTE["accent"] if k > 0.2 else viz.PALETTE["gray"] for k in kappas])
    ax.axhline(0.2, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=40, ha="right")
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Agreement vs humans")

    # empty top-right slot
    axes[0, 3].axis("off")

    fig.suptitle(f"Fig. 3a  Human-AI perception alignment on {args.model}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    viz.save_figure(fig, args.output, formats=("png", "pdf"))
    print(f"figure saved to {args.output}")


if __name__ == "__main__":
    main()
