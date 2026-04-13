"""Experiment 03 — produce Fig. 2 of the manuscript (urban vitality).

Fig. 2 shows:
 (a) OLS regression of AI-assigned livability against the five Jacobs attributes,
     with individual replicate lines in grey and the global fit in black.
 (b) Distribution of the fitted standardized coefficients (β) across the 100
     replicates, together with Jacobs' expected sign for each attribute.
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


_LIVABILITY_ALIASES = (
    "Livability", "Livability Score", "livability", "livability_score", "Score",
)


def _pick_livability(df: pd.DataFrame) -> str | None:
    for name in _LIVABILITY_ALIASES:
        if name in df.columns:
            return name
    for col in df.columns:
        if isinstance(col, str) and "livability" in col.lower():
            return col
    return None


def _load_replicates(model: str, scoring_prompt: str) -> list[pd.DataFrame]:
    in_dir = (config.paths.experiment_dir("vitality_scored")
              / model / scoring_prompt)
    frames: list[pd.DataFrame] = []
    for f in sorted(in_dir.rglob("run_*.xlsx")):  # recurse into batch sub-dirs
        try:
            df = pd.read_excel(f)
            y_col = _pick_livability(df)
            if y_col is None:
                continue
            # Normalise the column name so downstream panels always use
            # "Livability" regardless of how the replicate labelled it.
            if y_col != "Livability":
                df = df.rename(columns={y_col: "Livability"})
            present_attrs = [a for a in theories.VITALITY_ATTRIBUTES if a in df.columns]
            if not present_attrs:
                continue
            cols_to_num = [*present_attrs, "Livability"]
            for c in cols_to_num:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
                    errors="coerce",
                )
            frames.append(df.dropna(subset=cols_to_num))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(
            f"no usable replicates under {in_dir}. "
            f"run generate_attributes.py then score_livability.py first."
        )
    return frames


def _panel_a(ax, frames: list[pd.DataFrame]):
    """Scatter + per-attribute faceted regression lines."""
    attributes = list(theories.VITALITY_ATTRIBUTES)
    colors = plt.cm.tab10(np.linspace(0, 1, len(attributes)))

    for attr, color in zip(attributes, colors):
        xs_all, ys_all = [], []
        for df in frames:
            x = df[attr].values.astype(float)
            y = df["Livability"].values.astype(float)
            if len(x) < 5:
                continue
            xs_all.append(x)
            ys_all.append(y)
            # thin per-replicate fit line in the attribute's color
            try:
                m, b = np.polyfit(x, y, 1)
                xg = np.linspace(x.min(), x.max(), 30)
                ax.plot(xg, m * xg + b, color=color, alpha=0.1, linewidth=0.6)
            except np.linalg.LinAlgError:
                continue
        if xs_all:
            # Global aggregated fit line, thicker
            x_all = np.concatenate(xs_all)
            y_all = np.concatenate(ys_all)
            try:
                m, b = np.polyfit(x_all, y_all, 1)
                xg = np.linspace(x_all.min(), x_all.max(), 30)
                ax.plot(xg, m * xg + b, color=color, linewidth=2, label=attr)
            except np.linalg.LinAlgError:
                continue

    ax.set_xlabel("normalized attribute value")
    ax.set_ylabel("AI-assigned livability")
    ax.set_title("Per-attribute regression (global fits)")
    ax.legend(frameon=False, fontsize=8, loc="best")


def _panel_b(ax, frames: list[pd.DataFrame]):
    """Distribution of standardized β across replicates for each attribute."""
    attributes = list(theories.VITALITY_ATTRIBUTES)
    betas: dict[str, list[float]] = {a: [] for a in attributes}

    for df in frames:
        try:
            fit = fitting.fit_ols(df, "Livability", attributes)
        except ValueError:
            continue
        for a in attributes:
            betas[a].append(fit.coefficients[a])

    positions = np.arange(len(attributes))
    data = [betas[a] for a in attributes]
    bplot = ax.boxplot(data, positions=positions, widths=0.6,
                       patch_artist=True, showfliers=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(attributes)))
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    # expected-sign markers
    for i, a in enumerate(attributes):
        expected = theories.VITALITY_EXPECTED_SIGN[a]
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        marker = "▲" if expected > 0 else "▼"
        ax.text(i, max(max(betas[a]) if betas[a] else 0, 0) + 0.02,
                marker, ha="center", fontsize=12,
                color="green" if expected > 0 else "red")

    ax.set_xticks(positions)
    ax.set_xticklabels([a.replace(" ", "\n") for a in attributes], fontsize=9)
    ax.set_ylabel(r"standardized $\beta$")
    ax.set_title(f"β distribution across {len(frames)} replicates "
                 r"($\blacktriangle$ = Jacobs predicts positive)")


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--scoring-prompt", default=cfg["primary_scoring_prompt"])
    parser.add_argument("--output", type=Path,
                        default=config.paths.figures / "figure2.png")
    args = parser.parse_args(argv)

    viz.use_paper_style(font_size=12)
    frames = _load_replicates(args.model, args.scoring_prompt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _panel_a(axes[0], frames)
    _panel_b(axes[1], frames)

    fig.suptitle(
        f"Fig. 2  Urban vitality on {args.model} "
        f"(prompt: {args.scoring_prompt}, n={len(frames)} replicates)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    viz.save_figure(fig, args.output, formats=("png", "pdf"))
    print(f"figure saved to {args.output}")


if __name__ == "__main__":
    main()
