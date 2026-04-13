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


def aggregate(model: str, prompt: str, variant: str | None = None) -> pd.DataFrame:
    """Load and fit every replicate under the (model, prompt[, variant]) path.

    For the Supplementary Experiment 3 datasets the prompt slug
    ``distance_decay.city_150rings_2010`` is followed by a variant subfolder
    such as ``shanghai_1990_50km``; pass that name via ``variant``.
    Without ``variant``, the function searches the whole prompt tree
    recursively and groups replicates by their variant subdirectory
    automatically.
    """
    base = config.paths.model_dir("distance_decay", model, prompt)
    if variant:
        in_dir = base / variant
        files = sorted(in_dir.glob("run_*.xlsx"))
    else:
        in_dir = base
        files = sorted(base.rglob("run_*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No replicates under {in_dir}")

    rows = []
    for f in files:
        try:
            df = pd.read_excel(f)
            result = fit_one_replicate(df)
            result["replicate"] = int(f.stem.split("_")[-1])
            result["file"] = f.name
            # Store the variant path (relative to `base`) so the summary can
            # be grouped without re-aggregating.
            try:
                rel_parent = f.parent.relative_to(base).as_posix()
            except ValueError:
                rel_parent = ""
            result["variant"] = rel_parent or "_default_"
            rows.append(result)
        except Exception as e:
            log.warning("skipping %s: %s", f.name, e)
    return (pd.DataFrame(rows)
              .sort_values(["variant", "replicate"])
              .reset_index(drop=True))


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument(
        "--variant",
        default=None,
        help="For prompts that have per-city / per-year sub-experiments "
             "(e.g. distance_decay.city_150rings_2010), pass the sub-directory "
             "name such as 'shanghai_1990_50km'. Omit to aggregate every "
             "sub-variant under the prompt recursively.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    df = aggregate(args.model, args.prompt, variant=args.variant)
    print()
    print("=" * 72)
    print(f"Inverse-S fits — {args.model} / {args.prompt}"
          f"{' / ' + args.variant if args.variant else ''}")
    print("=" * 72)

    variants = df["variant"].unique() if "variant" in df.columns else ["_default_"]
    if len(variants) > 1:
        print(f"{'variant':<28s} {'n':>4s} {'r0':>8s} {'alpha':>8s} "
              f"{'beta':>8s} {'R²':>8s}")
        print("-" * 72)
        for v in sorted(variants):
            sub = df[df["variant"] == v]
            n_ok = sub["r_squared"].notna().sum()
            print(f"{v[:28]:<28s} {n_ok:>4d} "
                  f"{sub['r0'].mean():>8.2f} {sub['alpha'].mean():>8.2f} "
                  f"{sub['beta'].mean():>8.3f} {sub['r_squared'].mean():>8.3f}")
        print("-" * 72)
    print(f"Total replicates: {len(df)}, successful fits: "
          f"{df['r_squared'].notna().sum()}")
    print(f"Overall mean R² = {df['r_squared'].mean():.3f}")
    print("=" * 72)

    out = args.output or (
        config.paths.results / "distance_decay" / args.model / f"{args.prompt}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("summary → %s", out)


if __name__ == "__main__":
    main()
