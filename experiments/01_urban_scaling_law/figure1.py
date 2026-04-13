"""Experiment 01 — produce Fig. 1 of the manuscript.

Fig. 1 has two panels:

  a) Power-law fits for Zipf, infrastructure, and GDP scaling on the
     primary model (GPT-4o) with 95% CI bands derived from 100 replicates.
  b) Distribution of fitted scaling exponents β across the 100 replicates,
     compared to the empirical Bettencourt-2013 benchmark.

Run::

    python experiments/01_urban_scaling_law/figure1.py

By default the script reads replicates from
``data/generated/scaling_law/<primary_model>/<primary_prompt>/`` and writes
the figure to ``results/figures/figure1.png``.
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

from ai4us import config, fitting, theories, viz

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_replicates(model: str, prompt: str) -> list[pd.DataFrame]:
    in_dir = config.paths.model_dir("scaling_law", model, prompt)
    frames: list[pd.DataFrame] = []
    for f in sorted(in_dir.glob("run_*.xlsx")):
        try:
            df = pd.read_excel(f)
            for col in ("Population", "Infrastructure", "GDP"):
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
                    errors="coerce",
                )
            frames.append(df.dropna(subset=["Population", "Infrastructure", "GDP"]))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(
            f"No usable replicates found under {in_dir}. "
            f"Run `python experiments/01_urban_scaling_law/generate.py` first."
        )
    return frames


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _plot_power_law_panel(ax, frames: list[pd.DataFrame], x_col: str, y_col: str,
                          title: str, benchmark=None):
    """Draw one power-law panel with grey replicate lines + a mean fit."""
    betas = []
    for df in frames:
        x = df[x_col].values
        y = df[y_col].values
        try:
            fit = fitting.fit_power_law(x, y)
        except ValueError:
            continue
        betas.append(fit.beta)
        xs = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max()), 50)
        ax.plot(xs, fit.predict(xs), color=viz.PALETTE["gray"], alpha=0.1, linewidth=0.8)

    if betas:
        mean_beta = float(np.mean(betas))
        # reference line using mean exponent and median intercept
        all_x = np.concatenate([f[x_col].values for f in frames])
        all_y = np.concatenate([f[y_col].values for f in frames])
        master = fitting.fit_power_law(all_x, all_y)
        xs = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 50)
        ax.plot(xs, master.predict(xs), color=viz.PALETTE["accent"], linewidth=2.5,
                label=f"GenAI: β={mean_beta:.3f}")

    if benchmark is not None:
        # plot benchmark line anchored at the data midpoint
        all_x = np.concatenate([f[x_col].values for f in frames])
        all_y = np.concatenate([f[y_col].values for f in frames])
        # choose intercept so the line passes through (median x, median y)
        med_x = np.median(all_x)
        med_y = np.median(all_y)
        a = med_y / (med_x ** benchmark.beta)
        xs = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 50)
        ax.plot(xs, a * xs ** benchmark.beta, color=viz.PALETTE["empirical"],
                linestyle="--", linewidth=2,
                label=f"Bettencourt 2013: β={benchmark.beta:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right", fontsize=10)


def _plot_beta_distribution(ax, frames: list[pd.DataFrame]):
    """Panel (b): histograms of fitted β across replicates."""
    results = {"Zipf": [], "Infrastructure": [], "GDP": []}
    for df in frames:
        try:
            results["Zipf"].append(fitting.fit_zipf(df["Population"].values).beta)
        except ValueError:
            pass
        try:
            results["Infrastructure"].append(
                fitting.fit_power_law(df["Population"].values,
                                      df["Infrastructure"].values).beta
            )
        except ValueError:
            pass
        try:
            results["GDP"].append(
                fitting.fit_power_law(df["Population"].values,
                                      df["GDP"].values).beta
            )
        except ValueError:
            pass

    colors = [viz.PALETTE["accent"], viz.PALETTE["generated"], viz.PALETTE["theory"]]
    for (label, values), color in zip(results.items(), colors):
        if not values:
            continue
        ax.hist(values, bins=20, alpha=0.55, color=color, label=label, edgecolor="white")
        ax.axvline(float(np.mean(values)), color=color, linestyle="-", linewidth=1.5)

    # Benchmarks
    bench = theories.BETTENCOURT_2013
    ax.axvline(bench["infrastructure"].beta, color=viz.PALETTE["empirical"],
               linestyle="--", linewidth=1.5, label="Bettencourt 2013 (infra)")
    ax.axvline(bench["gdp"].beta, color=viz.PALETTE["empirical"],
               linestyle=":", linewidth=1.5, label="Bettencourt 2013 (GDP)")

    ax.set_xlabel(r"fitted exponent $\beta$")
    ax.set_ylabel("frequency")
    ax.set_title(f"Distribution of fitted β across {len(frames)} replicates")
    ax.legend(frameon=False, loc="upper right", fontsize=9)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument("--output", type=Path,
                        default=config.paths.figures / "figure1.png")
    args = parser.parse_args(argv)

    viz.use_paper_style(font_size=12)
    frames = _load_replicates(args.model, args.prompt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (a1, a2), (a3, a4) = axes

    # panel a1: Zipf (population vs rank) — we fake this by sorting populations
    # and using rank as x. Do it once here rather than in the helper because the
    # y_col handling differs.
    betas_zipf = []
    for df in frames:
        try:
            fit = fitting.fit_zipf(df["Population"].values)
            betas_zipf.append(fit.beta)
            sorted_pop = np.sort(df["Population"].values)[::-1]
            ranks = np.arange(1, len(sorted_pop) + 1)
            a1.plot(ranks, fit.predict(ranks),
                    color=viz.PALETTE["gray"], alpha=0.1, linewidth=0.8)
        except ValueError:
            continue
    if betas_zipf:
        a1.set_xscale("log"); a1.set_yscale("log")
        a1.set_xlabel("rank"); a1.set_ylabel("Population")
        a1.set_title(f"Zipf: mean β={np.mean(betas_zipf):.3f}")

    # panel a2: infrastructure scaling
    _plot_power_law_panel(a2, frames, "Population", "Infrastructure",
                          "Infrastructure scaling",
                          benchmark=theories.BETTENCOURT_2013["infrastructure"])

    # panel a3: GDP scaling
    _plot_power_law_panel(a3, frames, "Population", "GDP",
                          "GDP scaling",
                          benchmark=theories.BETTENCOURT_2013["gdp"])

    # panel a4: distribution of β
    _plot_beta_distribution(a4, frames)

    fig.suptitle(
        f"Fig. 1  Urban scaling law on {args.model} "
        f"(prompt: {args.prompt}, n={len(frames)} replicates)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    viz.save_figure(fig, args.output, formats=("png", "pdf"))
    print(f"figure saved to {args.output}")


if __name__ == "__main__":
    main()
