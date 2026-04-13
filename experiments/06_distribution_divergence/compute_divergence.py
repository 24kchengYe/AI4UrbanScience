"""Experiment 06 — compute MAE / Overlap Ratio / JSD between real and generated.

Replaces ``11datadifference.py`` and ``11datadifference2.py``.

For each scenario (R-G, R-GC, R-GA, R-R, G-G, GC-GC, GA-GA), the script
collects the relevant columns from the real and generated DataFrames and
computes three distributional metrics using :mod:`ai4us.metrics`.
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

from ai4us import config, metrics

log = logging.getLogger("divergence.compute")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _to_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
        errors="coerce",
    ).dropna().values


def _load_generated_pool(model: str, prompt: str, columns: list[str]) -> dict[str, np.ndarray]:
    in_dir = config.paths.model_dir("scaling_law", model, prompt)
    pools: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for f in sorted(in_dir.glob("run_*.xlsx")):
        try:
            df = pd.read_excel(f)
            for c in columns:
                if c in df.columns:
                    pools[c].append(_to_numeric(df[c]))
        except Exception as e:
            log.warning("skip %s: %s", f.name, e)
    return {c: np.concatenate(v) if v else np.empty(0) for c, v in pools.items()}


def _load_real(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {c: _to_numeric(df[c]) for c in columns if c in df.columns}


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--real-csv", type=Path,
                        default=config.paths.real_world / "chinese_cities.csv")
    parser.add_argument("--n-bins", type=int, default=cfg["n_bins"])
    parser.add_argument("--output", type=Path,
                        default=config.paths.results / "distribution_divergence" / "metrics.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    columns = cfg["columns"]

    if not args.real_csv.exists():
        raise SystemExit(
            f"real-world data missing at {args.real_csv}; run extract_real_data.py first"
        )
    real = _load_real(args.real_csv, columns)

    # Load every scenario once.
    scenarios = {name: _load_generated_pool(args.model, spec["prompt"], columns)
                 for name, spec in cfg["scenarios"].items()}

    rows: list[dict] = []
    for col in columns:
        r = real[col]
        if not len(r):
            log.warning("real has no %s values", col)
            continue
        for name, pool in scenarios.items():
            g = pool.get(col, np.empty(0))
            if not len(g):
                continue
            rows.append({
                "column": col,
                "comparison": f"R-{name}",
                "mae": metrics.mae_bin(r, g, n_bins=args.n_bins),
                "overlap_ratio": metrics.overlap_ratio(r, g, n_bins=args.n_bins),
                "jsd": metrics.jsd(r, g, n_bins=args.n_bins),
            })
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    log.info("metrics -> %s", args.output)


if __name__ == "__main__":
    main()
