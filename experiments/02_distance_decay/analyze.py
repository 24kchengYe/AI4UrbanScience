"""Experiment 02 — fit inverse-S decay across distance-decay replicates.

Replaces ``08AIUS-TheoryValid(distance).py`` and
``09AIUS-TheoryValidMultitime(distance).py`` and its visualization siblings.
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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai4us import config, fitting

log = logging.getLogger("distance_decay.analyze")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
        errors="coerce",
    )


def _circle_index(series: pd.Series) -> pd.Series:
    """Extract the ring number from strings like ``"Circle 42"`` or ``"42"``."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )


def fit_one_replicate(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["r"] = _circle_index(df["Circle Layer"])
    df["pop_density"] = _to_numeric(df["Population Density"])
    df["land_density"] = _to_numeric(df["Land Density"])
    df = df.dropna(subset=["r", "land_density"]).sort_values("r")

    out: dict = {"n_rows": len(df)}
    try:
        # Normalize land density to [0, 1] if values look like percentages.
        vals = df["land_density"].values.astype(float)
        if np.nanmax(vals) > 1.5:
            vals = vals / np.nanmax(vals)
        fit = fitting.fit_inverse_s(df["r"].values.astype(float), vals)
        out.update(r0=fit.r0, alpha=fit.alpha, beta=fit.beta, r_squared=fit.r_squared)
    except ValueError:
        out.update(r0=np.nan, alpha=np.nan, beta=np.nan, r_squared=np.nan)
    return out


def aggregate(model: str, prompt: str) -> pd.DataFrame:
    in_dir = config.paths.model_dir("distance_decay", model, prompt)
    files = sorted(in_dir.glob("run_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No replicates under {in_dir}")
    rows = []
    for f in files:
        try:
            df = pd.read_excel(f)
            result = fit_one_replicate(df)
            result["replicate"] = int(f.stem.split("_")[-1])
            result["file"] = f.name
            rows.append(result)
        except Exception as e:
            log.warning("skipping %s: %s", f.name, e)
    return pd.DataFrame(rows).sort_values("replicate").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    df = aggregate(args.model, args.prompt)
    print()
    print("=" * 60)
    print(f"Inverse-S fits across {len(df)} replicates")
    print(f"  mean r0       = {df['r0'].mean():.3f}")
    print(f"  mean alpha    = {df['alpha'].mean():.3f}")
    print(f"  mean beta     = {df['beta'].mean():.3f}")
    print(f"  mean R²       = {df['r_squared'].mean():.3f}")
    print(f"  successful    = {df['r_squared'].notna().sum()}/{len(df)}")
    print("=" * 60)

    out = args.output or (
        config.paths.results / "distance_decay" / args.model / f"{args.prompt}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("summary → %s", out)


if __name__ == "__main__":
    main()
