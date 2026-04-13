"""Experiment 06 — produce Fig. 4 of the manuscript.

Panel (a): overlaid frequency distributions (Population, GDP, Infrastructure)
           for real vs baseline-generated vs anchor-generated data.
Panel (b): pairwise JSD violin plots (R-R, R-G, G-G, ...).
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

from ai4us import config, metrics, viz

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _to_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
        errors="coerce",
    ).dropna().values


COLUMN_ALIASES = {
    "Infrastructure": ("Infrastructure", "Infra", "Infrastructure volume",
                        "Infrastructure Volume", "infra"),
    "Population":     ("Population",),
    "GDP":            ("GDP",),
}


def _resolve_col(df: pd.DataFrame, logical: str) -> str | None:
    for name in COLUMN_ALIASES.get(logical, (logical,)):
        if name in df.columns:
            return name
    return None


def _load_pool(model: str, prompt: str, columns: list[str]) -> dict[str, np.ndarray]:
    in_dir = config.paths.model_dir("scaling_law", model, prompt)
    out: dict[str, list] = {c: [] for c in columns}
    for f in sorted(in_dir.glob("run_*.xlsx")):
        try:
            df = pd.read_excel(f)
            for logical in columns:
                real_col = _resolve_col(df, logical)
                if real_col is not None:
                    out[logical].append(_to_numeric(df[real_col]))
        except Exception:
            pass
    return {c: np.concatenate(v) if v else np.empty(0) for c, v in out.items()}


def _panel_a(fig, axes, real, scenarios, columns):
    for ax, col in zip(axes, columns):
        r = real[col]
        # log-transform if dynamic range is large
        if r.size and np.nanmax(r) / max(np.nanmin(r[r > 0]), 1e-12) > 1e3:
            r_plot = np.log10(r[r > 0])
            transform = "log10"
        else:
            r_plot = r
            transform = ""
        ax.hist(r_plot, bins=30, density=True, alpha=0.55,
                color=viz.PALETTE["real"], label="Real (R)")
        for name, pool in scenarios.items():
            g = pool.get(col, np.empty(0))
            if not g.size:
                continue
            g_plot = np.log10(g[g > 0]) if transform == "log10" else g
            ax.hist(g_plot, bins=30, density=True, alpha=0.4,
                    label=f"Generated ({name})")
        ax.set_xlabel(f"{col} ({transform})" if transform else col)
        ax.set_ylabel("density")
        ax.set_title(col)
        ax.legend(frameon=False, fontsize=8)


def _panel_b(ax, real, scenarios, columns, n_bins):
    """Violin plot of pairwise JSD across columns."""
    labels = [f"R-{name}" for name in scenarios] + ["R-R", "G-G"]
    data: list[list[float]] = []
    for lbl in labels:
        values: list[float] = []
        for col in columns:
            r = real[col]
            if lbl == "R-R":
                if r.size >= 10:
                    half = len(r) // 2
                    values.append(metrics.jsd(r[:half], r[half:], n_bins=n_bins))
            elif lbl == "G-G":
                # compare baseline G pool against itself (split in half)
                g = scenarios["G"].get(col, np.empty(0))
                if g.size >= 10:
                    half = len(g) // 2
                    values.append(metrics.jsd(g[:half], g[half:], n_bins=n_bins))
            else:
                name = lbl.split("-", 1)[1]
                g = scenarios[name].get(col, np.empty(0))
                if r.size and g.size:
                    values.append(metrics.jsd(r, g, n_bins=n_bins))
        data.append(values or [0.0])

    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"],
                           [viz.PALETTE["accent"]] * len(scenarios)
                           + [viz.PALETTE["real"], viz.PALETTE["generated"]]):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Pairwise JSD")


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--real-csv", type=Path,
                        default=config.paths.real_world / "chinese_cities.csv")
    parser.add_argument("--output", type=Path,
                        default=config.paths.figures / "figure4.png")
    args = parser.parse_args(argv)

    viz.use_paper_style(font_size=11)
    columns = cfg["columns"]
    real_df = pd.read_csv(args.real_csv) if args.real_csv.exists() else pd.DataFrame()
    if real_df.empty:
        raise SystemExit(f"no real-world data at {args.real_csv}; "
                         f"run experiments/06_distribution_divergence/extract_real_data.py first")
    real = {}
    for logical in columns:
        real_col = _resolve_col(real_df, logical)
        if real_col is not None:
            real[logical] = _to_numeric(real_df[real_col])
    scenarios = {name: _load_pool(args.model, spec["prompt"], columns)
                 for name, spec in cfg["scenarios"].items()}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _panel_a(fig, axes[0].tolist() + axes[1, :1].tolist(),
             real, scenarios, columns[:3])
    _panel_b(axes[1, 1], real, scenarios, columns, cfg["n_bins"])
    fig.suptitle(f"Fig. 4  Data distribution divergence ({args.model})",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    viz.save_figure(fig, args.output, formats=("png", "pdf"))
    print(f"figure saved to {args.output}")


if __name__ == "__main__":
    main()
