from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from ..source_data import SOURCES
from .style import apply_style, color_for, draw_interval, panel_title, save_bundle, short_label


SOURCE = SOURCES["Fig03_data"]


def _forest(axis, data: pd.DataFrame, letter: str, title: str, *, log_scale: bool = False, reference: float | None = None) -> None:
    label_columns = [column for column in ("metric", "variable") if column in data]
    data = data.copy()
    data["row_label"] = data[label_columns].fillna("").astype(str).agg(" | ".join, axis=1).str.strip(" |")
    labels = list(dict.fromkeys(data["row_label"].tolist()))
    conditions = list(dict.fromkeys(data["condition"].tolist()))
    offsets = np.linspace(0.24, -0.24, len(conditions)) if len(conditions) > 1 else [0.0]
    for condition, offset in zip(conditions, offsets):
        subset = data[data["condition"] == condition].set_index("row_label")
        for index, label in enumerate(labels):
            if label not in subset.index:
                continue
            row = subset.loc[label]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            estimate = float(row["estimate"])
            low = pd.to_numeric(pd.Series([row.get("ci_low", row.get("q025", np.nan))]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([row.get("ci_high", row.get("q975", np.nan))]), errors="coerce").iloc[0]
            draw_interval(axis, estimate, index + offset, float(low), float(high), color=color_for(condition), markerfacecolor="white", label=condition if index == 0 else None)
    axis.set_yticks(range(len(labels)), [short_label(item, 42) for item in labels])
    if reference is not None:
        axis.axvline(reference, color="#777777", linestyle="--", linewidth=0.65)
    if log_scale:
        axis.set_xscale("log")
        axis.set_xlim(0.25, 9.0)
        axis.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0, 2.0, 5.0]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_locator(NullLocator())
    axis.legend(frameon=False, fontsize=5.7, loc="best")
    panel_title(axis, letter, title)


def render(output_dir: Path) -> dict:
    apply_style()
    frame = pd.read_csv(SOURCE)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.05))
    figure.subplots_adjust(left=0.18, right=0.98, top=0.94, bottom=0.09, wspace=0.58, hspace=0.48)
    _forest(axes[0, 0], frame[frame["benchmark"] == "Urban Scaling"], "a", "Urban Scaling")
    _forest(axes[0, 1], frame[frame["benchmark"] == "Distance Decay"], "b", "Distance Decay")
    _forest(axes[1, 0], frame[frame["benchmark"] == "Vitality"], "c", "Jacobs-informed Vitality", log_scale=True, reference=1.0)
    _forest(axes[1, 1], frame[frame["benchmark"] == "Perception"], "d", "Place Pulse Perception")
    return save_bundle(figure, output_dir / "figure3_empirical_variation_public_rerender")
