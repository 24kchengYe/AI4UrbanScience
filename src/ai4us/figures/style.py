from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Baseline": "#8C8C8C",
    "Baseline prompt": "#8C8C8C",
    "BASE": "#8C8C8C",
    "Blueprint": "#D97706",
    "Blueprint prompt": "#D97706",
    "Full Blueprint": "#D97706",
    "FULL": "#D97706",
    "Empirical": "#2F6B8A",
    "Empirical reference": "#2F6B8A",
    "52-site empirical": "#2F6B8A",
    "Published 50-city parameters": "#2F6B8A",
    "Scaling": "#44546A",
    "Urban Scaling": "#44546A",
    "Distance Decay": "#C45A35",
    "Vitality": "#2E7D5B",
    "Perception": "#7656A3",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def color_for(label: str) -> str:
    if label in COLORS:
        return COLORS[label]
    normalized = label.casefold()
    if "baseline" in normalized:
        return COLORS["Baseline"]
    if "blueprint" in normalized:
        return COLORS["Blueprint"]
    if "empirical" in normalized:
        return COLORS["Empirical"]
    return "#4B5563"


def panel_title(axis, letter: str, title: str) -> None:
    axis.text(
        -0.12,
        1.05,
        letter,
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axis.text(
        -0.03,
        1.05,
        title,
        transform=axis.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
    )


def numeric(values) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def draw_interval(axis, x: float, y: float, low: float, high: float, **kwargs) -> None:
    if np.isfinite(low) and np.isfinite(high):
        axis.errorbar(
            x,
            y,
            xerr=[[x - low], [high - x]],
            fmt="o",
            capsize=1.7,
            linewidth=0.8,
            markersize=3.4,
            **kwargs,
        )
    else:
        axis.plot(x, y, "o", markersize=3.4, **kwargs)


def save_bundle(figure, output_base: Path) -> dict[str, dict[str, str | int]]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str | int]] = {}
    for suffix in ("pdf", "svg", "png"):
        path = output_base.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=600, facecolor="white", bbox_inches="tight")
        outputs[suffix] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    plt.close(figure)
    return outputs


def short_label(value: object, limit: int = 34) -> str:
    text = str(value).replace("_", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
