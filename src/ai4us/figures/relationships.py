from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..source_data import SOURCES
from .style import apply_style, color_for, draw_interval, panel_title, save_bundle


SOURCE = SOURCES["Fig02_data"]


def _scaling(axis, frame: pd.DataFrame) -> None:
    data = frame[(frame["benchmark"] == "Urban Scaling") & (frame["record_type"] == "fit_summary")]
    relations = ["Road", "GDP", "Zipf"]
    conditions = ["BASE", "FULL"]
    offsets = {"BASE": 0.13, "FULL": -0.13}
    for condition in conditions:
        subset = data[data["condition"] == condition].set_index("relation")
        for index, relation in enumerate(relations):
            row = subset.loc[relation]
            draw_interval(
                axis,
                float(row["estimate"]),
                index + offsets[condition],
                float(row["ci_low"]),
                float(row["ci_high"]),
                color=color_for(condition),
                markerfacecolor="white",
                label=("Baseline" if condition == "BASE" else "Blueprint") if index == 0 else None,
            )
    axis.axvline(0, color="#B8B8B8", linewidth=0.6)
    axis.set_yticks(range(len(relations)), relations)
    axis.set_xlabel("Table-level exponent, mean and 2.5–97.5% range")
    axis.legend(frameon=False, loc="lower right")
    panel_title(axis, "a", "Urban Scaling")


def _distance_decay(axis, frame: pd.DataFrame) -> None:
    data = frame[frame["benchmark"] == "Distance Decay"].copy()
    data = data[data["parameter"].astype(str).str.len() > 0]
    parameters = ["alpha_hat", "c_hat", "D_hat_km"]
    labels = ["Steepness α", "Peripheral density c", "Radial scale D"]
    conditions = ["Published 50-city parameters", "Baseline", "Blueprint"]
    offsets = dict(zip(conditions, [0.22, 0.0, -0.22]))
    for condition in conditions:
        subset = data[data["condition"] == condition].set_index("parameter")
        for index, parameter in enumerate(parameters):
            row = subset.loc[parameter]
            draw_interval(
                axis,
                float(row["estimate"]),
                index + offsets[condition],
                float(row["ci_low"]),
                float(row["ci_high"]),
                color=color_for(condition),
                markerfacecolor="white",
                label=condition.replace("Published 50-city parameters", "Empirical") if index == 0 else None,
            )
    axis.set_xscale("log")
    axis.set_yticks(range(3), labels)
    axis.set_xlabel("Parameter value (log scale)")
    axis.legend(frameon=False, loc="lower right")
    panel_title(axis, "b", "Distance Decay")


def _vitality(axis, frame: pd.DataFrame) -> None:
    data = frame[frame["benchmark"] == "Vitality"].copy()
    features = list(dict.fromkeys(data["feature"].tolist()))
    conditions = ["52-site empirical", "Baseline", "Full Blueprint"]
    offsets = dict(zip(conditions, [0.22, 0.0, -0.22]))
    for condition in conditions:
        subset = data[data["condition"] == condition].set_index("feature")
        for index, feature in enumerate(features):
            row = subset.loc[feature]
            draw_interval(
                axis,
                float(row["estimate"]),
                index + offsets[condition],
                float(row["ci_low"]),
                float(row["ci_high"]),
                color=color_for(condition),
                markerfacecolor="white",
                label=condition.replace("52-site empirical", "Empirical").replace("Full Blueprint", "Blueprint") if index == 0 else None,
            )
    axis.axvline(0, color="#777777", linestyle="--", linewidth=0.6)
    axis.set_yticks(range(len(features)), features)
    axis.set_xlabel("Standardized coefficient")
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    panel_title(axis, "c", "Jacobs-informed Vitality")


def _perception(axis, frame: pd.DataFrame) -> None:
    axis.set_axis_off()
    data = frame[frame["benchmark"] == "Perception"].copy()
    dimensions = ["beautiful", "wealthy", "safety", "depressing", "lively", "boring"]
    metrics = ["Three-class accuracy", "Three-class Cohen kappa"]
    labels = ["Accuracy", "Cohen's κ"]
    for position, (metric, label) in enumerate(zip(metrics, labels)):
        inset = axis.inset_axes([position * 0.53, 0.04, 0.47, 0.88])
        subset = data[data["metric"] == metric].set_index("condition").loc[dimensions]
        for y, (_, row) in enumerate(subset.iterrows()):
            draw_interval(
                inset,
                float(row["estimate"]),
                y,
                float(row["ci_low"]),
                float(row["ci_high"]),
                color=color_for("Perception"),
                markerfacecolor="white",
            )
        inset.set_yticks(range(6), [item.title() for item in dimensions] if position == 0 else [])
        inset.set_xlabel(label)
        if position == 1:
            inset.axvline(0, color="#777777", linestyle="--", linewidth=0.6)
    panel_title(axis, "d", "Place Pulse Perception")


def render(output_dir: Path) -> dict:
    apply_style()
    frame = pd.read_csv(SOURCE)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.65))
    figure.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.12, wspace=0.48, hspace=0.48)
    _scaling(axes[0, 0], frame)
    _distance_decay(axes[0, 1], frame)
    _vitality(axes[1, 0], frame)
    _perception(axes[1, 1], frame)
    return save_bundle(figure, output_dir / "figure2_relationships_public_rerender")

