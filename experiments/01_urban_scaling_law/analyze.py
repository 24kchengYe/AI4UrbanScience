"""Experiment 01 — aggregate scaling-law analysis across replicates.

For a given (model, prompt) combination this script:

1. Loads every replicate file under
   ``data/generated/scaling_law/<model>/<prompt>/run_*.xlsx``
2. Fits Zipf, infrastructure-scaling, and GDP-scaling on each replicate.
3. Writes a summary CSV with per-replicate exponents and R².
4. Optionally prints a benchmark comparison against Bettencourt (2013).

The output CSV is consumed by ``figure1.py`` (and by ``sensitivity.py``
when it needs an aggregated view).
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

log = logging.getLogger("scaling_law.analyze")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Per-replicate fit
# ---------------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column that might contain strings with extra symbols."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
        errors="coerce",
    )


def fit_one_replicate(df: pd.DataFrame) -> dict:
    """Run all three fits on a single replicate. Returns a flat dict."""
    df = df.copy()
    df["Population"] = _to_numeric(df["Population"])
    df["Infrastructure"] = _to_numeric(df["Infrastructure"])
    df["GDP"] = _to_numeric(df["GDP"])
    df = df.dropna(subset=["Population", "Infrastructure", "GDP"])

    out: dict = {"n_rows": len(df)}

    # Zipf: population vs rank
    try:
        zipf = fitting.fit_zipf(df["Population"].values)
        out.update(zipf_beta=zipf.beta, zipf_r2=zipf.r_squared)
    except ValueError as e:
        log.debug("zipf fit failed: %s", e)
        out.update(zipf_beta=np.nan, zipf_r2=np.nan)

    # Infra scaling: infrastructure ~ population^beta
    try:
        fit = fitting.fit_power_law(df["Population"].values,
                                    df["Infrastructure"].values)
        out.update(infra_beta=fit.beta, infra_r2=fit.r_squared)
    except ValueError:
        out.update(infra_beta=np.nan, infra_r2=np.nan)

    # GDP scaling
    try:
        fit = fitting.fit_power_law(df["Population"].values,
                                    df["GDP"].values)
        out.update(gdp_beta=fit.beta, gdp_r2=fit.r_squared)
    except ValueError:
        out.update(gdp_beta=np.nan, gdp_r2=np.nan)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(model: str, prompt: str) -> pd.DataFrame:
    """Load and fit every replicate under ``data/generated/scaling_law/<model>/<prompt>/``."""
    in_dir = config.paths.model_dir("scaling_law", model, prompt)
    files = sorted(in_dir.glob("run_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No replicate files found in {in_dir}")

    records: list[dict] = []
    for f in files:
        try:
            df = pd.read_excel(f)
            fit = fit_one_replicate(df)
            fit["replicate"] = int(f.stem.split("_")[-1])
            fit["file"] = f.name
            records.append(fit)
        except Exception as e:
            log.warning("skipping %s: %s", f.name, e)

    return pd.DataFrame(records).sort_values("replicate").reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    bench = theories.BETTENCOURT_2013
    mean_r2 = df[["zipf_r2", "infra_r2", "gdp_r2"]].mean()
    mean_beta = df[["zipf_beta", "infra_beta", "gdp_beta"]].mean()
    print()
    print("=" * 62)
    print(f"{'metric':<15} {'mean beta':>12} {'mean R²':>10} {'benchmark':>20}")
    print("-" * 62)
    print(f"{'Zipf':<15} {mean_beta['zipf_beta']:>12.3f} {mean_r2['zipf_r2']:>10.3f}"
          f" {'(theoretical -1)':>20}")
    print(f"{'Infrastructure':<15} {mean_beta['infra_beta']:>12.3f} {mean_r2['infra_r2']:>10.3f}"
          f" {f'{bench['infrastructure'].beta:.3f} ± {bench['infrastructure'].beta_sd:.3f}':>20}")
    print(f"{'GDP':<15} {mean_beta['gdp_beta']:>12.3f} {mean_r2['gdp_r2']:>10.3f}"
          f" {f'{bench['gdp'].beta:.3f} ± {bench['gdp'].beta_sd:.3f}':>20}")
    print("=" * 62)
    print(f"Successful replicates: {df['zipf_beta'].notna().sum()}/{len(df)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the summary CSV "
                             "(default: results/scaling_law/<model>/<prompt>.csv)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = aggregate(args.model, args.prompt)
    print_summary(df)

    out = args.output or (
        config.paths.results / "scaling_law" / args.model / f"{args.prompt}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
