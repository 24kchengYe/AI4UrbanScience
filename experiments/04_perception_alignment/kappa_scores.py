"""Experiment 04 — compute Cohen's Kappa between AI and human choices.

Supports two input shapes:

1. **Per-dimension** CSVs produced by :mod:`pairwise_eval`, layout:
   ``data/generated/perception_alignment/<model>/<dimension>.csv``
   with columns ``human_choice, ai_choice``.

2. **Unified** CSVs produced by the research-time pipeline, layout:
   ``data/generated/perception_alignment/correlational/ai_vs_human_validation/ai_vs_human_choices_unified_*.csv``
   with columns ``category, human_winner, ai_winner`` (one row per pair).

The script auto-detects the format and computes per-dimension Kappa +
agreement rates. The output is the same whichever format was on disk.
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

import pandas as pd
import yaml

from ai4us import config

log = logging.getLogger("perception.kappa")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------

_LABELS_UPPER = ["LEFT", "RIGHT", "EQUAL"]


def _normalise(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def cohens_kappa(human: pd.Series, ai: pd.Series) -> dict:
    """Compute Cohen's Kappa between two categorical series over LEFT/RIGHT/EQUAL."""
    h = _normalise(human)
    a = _normalise(ai)
    valid = h.isin(_LABELS_UPPER) & a.isin(_LABELS_UPPER)
    h, a = h[valid], a[valid]
    n = len(h)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    po = float((h == a).mean())
    pe = sum((h == lab).mean() * (a == lab).mean() for lab in _LABELS_UPPER)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": int(n), "agreement": po, "kappa": float(kappa)}


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------

def _find_per_dim_inputs(model: str, dimensions: list[str]) -> dict[str, pd.DataFrame]:
    """Look for one CSV per dimension produced by pairwise_eval."""
    in_dir = config.paths.experiment_dir("perception_alignment") / model
    if not in_dir.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for dim in dimensions:
        f = in_dir / f"{dim}.csv"
        if f.exists():
            df = pd.read_csv(f)
            # Rename to canonical columns if needed
            if "human_choice" in df.columns and "ai_choice" in df.columns:
                out[dim] = df.rename(
                    columns={"human_choice": "human", "ai_choice": "ai"}
                )
    return out


def _find_unified_input() -> pd.DataFrame | None:
    """Look for the research-time unified CSV with category / human_winner / ai_winner."""
    root = config.paths.experiment_dir("perception_alignment")
    candidates = sorted(root.rglob("ai_vs_human_choices_unified*.csv"))
    if not candidates:
        return None
    # Pick the first match (if several variants exist, the user can inspect them
    # individually via the --input flag below).
    return pd.read_csv(candidates[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to a unified CSV (category / human_winner / ai_winner). "
             "Overrides auto-detection.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dimensions = cfg["dimensions"]
    rows: list[dict] = []

    # ---- 1) Explicit --input overrides everything ----
    if args.input is not None:
        unified = pd.read_csv(args.input)
        for dim in dimensions:
            sub = unified[unified["category"].str.lower() == dim.lower()]
            res = cohens_kappa(sub["human_winner"], sub["ai_winner"])
            res["dimension"] = dim
            rows.append(res)

    # ---- 2) Try per-dimension inputs from pairwise_eval.py ----
    elif _find_per_dim_inputs(args.model, dimensions):
        per_dim = _find_per_dim_inputs(args.model, dimensions)
        for dim in dimensions:
            if dim not in per_dim:
                log.warning("missing per-dim input for %s", dim)
                continue
            df = per_dim[dim]
            res = cohens_kappa(df["human"], df["ai"])
            res["dimension"] = dim
            rows.append(res)

    # ---- 3) Fall back to the research-time unified CSV ----
    else:
        unified = _find_unified_input()
        if unified is None:
            raise FileNotFoundError(
                "No kappa input found. Expected either\n"
                f"  data/generated/perception_alignment/{args.model}/<dim>.csv\n"
                "(output of pairwise_eval.py), or\n"
                "  data/generated/perception_alignment/correlational/"
                "ai_vs_human_validation/ai_vs_human_choices_unified*.csv\n"
                "from the research-time pipeline."
            )
        for dim in dimensions:
            sub = unified[unified["category"].str.lower() == dim.lower()]
            res = cohens_kappa(sub["human_winner"], sub["ai_winner"])
            res["dimension"] = dim
            rows.append(res)

    result = pd.DataFrame(rows, columns=["dimension", "n", "agreement", "kappa"])

    print()
    print("=" * 64)
    print(f"Human-AI agreement — {args.model}")
    print("-" * 64)
    for _, r in result.iterrows():
        print(f"  {r['dimension']:<12}  n={int(r['n']):>5d}  "
              f"agreement={r['agreement']:.3f}  κ={r['kappa']:.3f}")
    print("-" * 64)
    if not result.empty:
        print(f"  mean κ: {result['kappa'].mean():.3f}")
    print("=" * 64)

    out = args.output or (
        config.paths.results / "perception_alignment" / f"{args.model}_kappa.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    log.info("summary -> %s", out)


if __name__ == "__main__":
    main()
