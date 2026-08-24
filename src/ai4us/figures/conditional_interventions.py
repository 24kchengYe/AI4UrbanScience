from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..source_data import SOURCES
from .style import apply_style, color_for, draw_interval, panel_title, save_bundle, short_label


SOURCE = SOURCES["Fig04_data"]


def _panel(axis, data: pd.DataFrame, letter: str, title: str) -> None:
    data = data.reset_index(drop=True)
    for index, row in data.iterrows():
        estimate = float(row["estimate"])
        low = pd.to_numeric(pd.Series([row.get("ci_low", np.nan)]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([row.get("ci_high", np.nan)]), errors="coerce").iloc[0]
        draw_interval(axis, estimate, index, float(low), float(high), color=color_for(title.split(" |")[0]), markerfacecolor="white")
    labels = [short_label(value, 38) for value in data["contrast"].tolist()]
    axis.set_yticks(range(len(labels)), labels)
    if title.startswith("Vitality"):
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=0.65)
    elif title.startswith("Distance"):
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=0.65)
    else:
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.65)
    axis.set_xlabel("Estimated response with 95% interval where available")
    panel_title(axis, letter, title)


def render(output_dir: Path) -> dict:
    apply_style()
    frame = pd.read_csv(SOURCE)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    figure.subplots_adjust(left=0.19, right=0.98, top=0.94, bottom=0.10, wspace=0.56, hspace=0.46)
    definitions = [
        ("Urban Scaling", "a", "Urban Scaling | Population"),
        ("Distance Decay", "b", "Distance Decay | Distance"),
        ("Vitality", "c", "Vitality | Functional Mix"),
        ("Perception", "d", "Perception | Theme edits"),
    ]
    for axis, (benchmark, letter, title) in zip(axes.flat, definitions):
        _panel(axis, frame[frame["benchmark"] == benchmark], letter, title)
    return save_bundle(figure, output_dir / "figure4_conditional_interventions_public_rerender")

