"""Experiment 07 — Optimal-Transport calibration of generated city data.

Collapses the five legacy calibration scripts into a single module with
three functions:

* :func:`quantile_match`   – simple per-column quantile alignment
* :func:`ot_marginal`      – 1-D Sinkhorn OT applied column-by-column
* :func:`ot_joint`         – multidimensional OT over all columns jointly

All three take the same ``(real, generated)`` DataFrame inputs and return
calibrated generated values ready to be re-fitted with
:mod:`ai4us.fitting`.

Replaces ``13生成数据分布矫正.py``, ``14消融实验 llm+OT.py``,
``15标度系数验证 llm+OT.py``, ``16不同方法的数据规律拟合可视化.py``,
``16真实数据校准SCforAI4US.py``.
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

try:
    import ot  # type: ignore
except ImportError:  # pragma: no cover
    ot = None

from ai4us import config, fitting

log = logging.getLogger("calibration_ot")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------

def quantile_match(real: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Map each generated point to the real-data quantile with the same rank."""
    g_sorted_idx = np.argsort(generated)
    ranks = np.empty_like(g_sorted_idx)
    ranks[g_sorted_idx] = np.arange(len(generated))
    target = np.sort(real)
    if len(target) == 0:
        return generated
    # Map rank proportion to target quantile
    q = ranks / max(len(generated) - 1, 1)
    return np.interp(q, np.linspace(0, 1, len(target)), target)


def ot_marginal(real: np.ndarray, generated: np.ndarray,
                reg: float = 0.05, max_iter: int = 2000) -> np.ndarray:
    """1-D Sinkhorn OT column-by-column.

    Returns a new array of the same length as ``generated`` whose per-point
    values have been pulled toward the empirical distribution of ``real``.
    """
    if ot is None:
        log.warning("POT not installed; falling back to quantile_match")
        return quantile_match(real, generated)

    a = np.ones(len(generated)) / len(generated)
    b = np.ones(len(real)) / len(real)
    # Cost matrix = squared Euclidean distance between each pair of values
    M = (generated.reshape(-1, 1) - real.reshape(1, -1)) ** 2
    M /= (M.max() or 1.0)
    T = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)
    # Barycentric projection = each generated sample -> expected real value
    return (T @ real) * len(generated)


def ot_joint(real: pd.DataFrame, generated: pd.DataFrame,
             reg: float = 0.05, max_iter: int = 2000) -> pd.DataFrame:
    """Joint OT over all columns of ``generated`` simultaneously."""
    if ot is None:
        log.warning("POT not installed; falling back to per-column quantile_match")
        return pd.DataFrame({c: quantile_match(real[c].values, generated[c].values)
                             for c in generated.columns})

    R = real.values.astype(float)
    G = generated.values.astype(float)
    a = np.ones(len(G)) / len(G)
    b = np.ones(len(R)) / len(R)
    M = ot.dist(G, R, metric="euclidean")
    M /= (M.max() or 1.0)
    T = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)
    calibrated = (T @ R) * len(G)
    return pd.DataFrame(calibrated, columns=generated.columns)


# ---------------------------------------------------------------------------
# Driver: compute scaling-exponent shifts under each method
# ---------------------------------------------------------------------------

def run_ablation(real: pd.DataFrame, generated: pd.DataFrame,
                 columns: list[str], methods: list[str],
                 ot_reg: float, ot_iter: int) -> pd.DataFrame:
    rows: list[dict] = []
    for method in methods:
        if method == "raw":
            calibrated = generated[columns].copy()
        elif method == "quantile_match":
            calibrated = pd.DataFrame(
                {c: quantile_match(real[c].values, generated[c].values) for c in columns}
            )
        elif method == "ot_marginal":
            calibrated = pd.DataFrame(
                {c: ot_marginal(real[c].values, generated[c].values,
                                reg=ot_reg, max_iter=ot_iter) for c in columns}
            )
        elif method == "ot_joint":
            calibrated = ot_joint(real[columns], generated[columns],
                                  reg=ot_reg, max_iter=ot_iter)
        else:
            log.warning("unknown method %s", method)
            continue

        # Re-fit scaling law after calibration
        try:
            infra_fit = fitting.fit_power_law(
                calibrated["Population"].values, calibrated["Infrastructure"].values
            )
            gdp_fit = fitting.fit_power_law(
                calibrated["Population"].values, calibrated["GDP"].values
            )
        except ValueError as e:
            log.warning("fit failed for %s: %s", method, e)
            continue
        rows.append({
            "method": method,
            "infra_beta": infra_fit.beta,
            "infra_r2": infra_fit.r_squared,
            "gdp_beta": gdp_fit.beta,
            "gdp_r2": gdp_fit.r_squared,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument("--real-csv", type=Path,
                        default=config.paths.real_world / "chinese_cities.csv")
    parser.add_argument("--output", type=Path,
                        default=config.paths.results / "calibration_ot" / "ablation.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    columns = cfg["columns"]
    real = pd.read_csv(args.real_csv)
    real_num = real[columns].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()

    # Aggregate all generated replicates into one big table
    gen_dir = config.paths.model_dir("scaling_law", args.model, args.prompt)
    frames: list[pd.DataFrame] = []
    for f in sorted(gen_dir.glob("run_*.xlsx")):
        try:
            df = pd.read_excel(f)
            for c in columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
                    errors="coerce",
                )
            frames.append(df[columns].dropna())
        except Exception:
            pass
    if not frames:
        raise SystemExit(f"no replicates in {gen_dir}")
    gen_all = pd.concat(frames, ignore_index=True)

    result = run_ablation(
        real_num, gen_all, columns, cfg["methods"],
        cfg["ot"]["regularization"], cfg["ot"]["max_iterations"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    log.info("ablation table -> %s", args.output)


if __name__ == "__main__":
    main()
