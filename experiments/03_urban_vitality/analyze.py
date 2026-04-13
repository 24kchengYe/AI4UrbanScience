"""Experiment 03 — aggregate OLS fits of Jacobs' urban vitality model.

For each replicate of scored blocks we fit::

    Livability = a + b1 * PopDensity + b2 * BuildingMix + ... + b5 * TallBuilding

then report the mean coefficient for each attribute together with the
expected sign from Jacobs' theory (see :data:`ai4us.theories.VITALITY_EXPECTED_SIGN`).
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

from ai4us import config, fitting, theories

log = logging.getLogger("vitality.analyze")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(
            out[c].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
            errors="coerce",
        )
    return out


def fit_replicate(df: pd.DataFrame, attributes: list[str]) -> dict | None:
    df = _numeric(df, [*attributes, "Livability"]).dropna(subset=[*attributes, "Livability"])
    if len(df) < len(attributes) + 5:
        return None
    fit = fitting.fit_ols(df, "Livability", attributes)
    row: dict = {"n_rows": fit.n, "r_squared": fit.r_squared, "intercept": fit.intercept}
    for attr, b in fit.coefficients.items():
        row[f"beta__{attr}"] = b
        row[f"p__{attr}"] = fit.p_values[attr]
    return row


def aggregate(model: str, scoring_prompt: str,
              attributes: list[str]) -> pd.DataFrame:
    in_dir = (config.paths.experiment_dir("vitality_scored")
              / model / scoring_prompt)
    files = sorted(in_dir.glob("run_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No scored replicates under {in_dir}")
    rows = []
    for f in files:
        try:
            result = fit_replicate(pd.read_excel(f), attributes)
            if result is None:
                log.warning("skipping %s (insufficient data)", f.name)
                continue
            result["replicate"] = int(f.stem.split("_")[-1])
            rows.append(result)
        except Exception as e:
            log.warning("skipping %s: %s", f.name, e)
    return pd.DataFrame(rows).sort_values("replicate").reset_index(drop=True)


def print_summary(df: pd.DataFrame, attributes: list[str]) -> None:
    print()
    print("=" * 80)
    print(f"OLS vitality fits across {len(df)} replicates  (mean R² = {df['r_squared'].mean():.3f})")
    print("-" * 80)
    print(f"{'attribute':<22} {'mean β':>12} {'p(mean β)':>14} {'expected sign':>15} {'matches':>10}")
    print("-" * 80)
    for attr in attributes:
        beta_col = f"beta__{attr}"
        p_col = f"p__{attr}"
        mean_beta = df[beta_col].mean()
        mean_p = df[p_col].mean()
        expected = theories.VITALITY_EXPECTED_SIGN.get(attr, 0)
        matches = "✓" if np.sign(mean_beta) == expected else "✗"
        print(f"{attr:<22} {mean_beta:>12.4f} {mean_p:>14.4g} "
              f"{'+' if expected > 0 else '−':>15} {matches:>10}")
    print("=" * 80)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--scoring-prompt", default=cfg["primary_scoring_prompt"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    attributes = list(theories.VITALITY_ATTRIBUTES)
    df = aggregate(args.model, args.scoring_prompt, attributes)
    print_summary(df, attributes)

    out = args.output or (
        config.paths.results / "vitality" / args.model / f"{args.scoring_prompt}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("summary -> %s", out)


if __name__ == "__main__":
    main()
