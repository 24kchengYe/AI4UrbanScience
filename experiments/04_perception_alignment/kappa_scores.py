"""Experiment 04 — compute Cohen's Kappa between AI and human choices.

Reads the per-dimension CSVs produced by :mod:`pairwise_eval` and reports
per-dimension and overall agreement rates and Kappa coefficients.
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

log = logging.getLogger("perception.kappa")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cohens_kappa(human: pd.Series, ai: pd.Series) -> dict:
    """Compute Cohen's Kappa between two categorical series over LEFT/RIGHT/EQUAL."""
    valid = human.isin(["LEFT", "RIGHT", "EQUAL"]) & ai.isin(["LEFT", "RIGHT", "EQUAL"])
    h, a = human[valid], ai[valid]
    n = len(h)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    po = float((h == a).mean())
    labels = ["LEFT", "RIGHT", "EQUAL"]
    pe = 0.0
    for lab in labels:
        pe += (h == lab).mean() * (a == lab).mean()
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": int(n), "agreement": po, "kappa": float(kappa)}


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    in_dir = config.paths.experiment_dir("perception_alignment") / args.model
    if not in_dir.exists():
        raise FileNotFoundError(f"No judgments found at {in_dir}; run pairwise_eval.py first")

    rows: list[dict] = []
    for dim in cfg["dimensions"]:
        f = in_dir / f"{dim}.csv"
        if not f.exists():
            log.warning("missing %s", f)
            continue
        df = pd.read_csv(f)
        res = cohens_kappa(df["human_choice"], df["ai_choice"])
        res["dimension"] = dim
        rows.append(res)

    result = pd.DataFrame(rows, columns=["dimension", "n", "agreement", "kappa"])
    print()
    print("=" * 60)
    print(f"Human-AI agreement for {args.model}")
    print("-" * 60)
    for _, r in result.iterrows():
        print(f"  {r['dimension']:<12}  n={r['n']:>5}  "
              f"agreement={r['agreement']:.3f}  κ={r['kappa']:.3f}")
    print("-" * 60)
    print(f"  mean κ: {result['kappa'].mean():.3f}")
    print("=" * 60)

    out = args.output or (
        config.paths.results / "perception_alignment" / f"{args.model}_kappa.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    log.info("summary -> %s", out)


if __name__ == "__main__":
    main()
