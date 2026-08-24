from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..source_data import SOURCES
from .style import apply_style, color_for, draw_interval, panel_title, save_bundle, short_label


SOURCE = SOURCES["Fig05_data"]


def _behaviour(axis, frame: pd.DataFrame) -> None:
    data = frame[frame["panel"] == "a"].copy()
    for theory, group in data.groupby("theory", sort=False):
        axis.scatter(
            pd.to_numeric(group["closed_value"]),
            pd.to_numeric(group["open_value"]),
            s=12,
            alpha=0.74,
            edgecolor="white",
            linewidth=0.25,
            color=color_for(theory),
            label=theory,
        )
    values = pd.concat(
        [pd.to_numeric(data["closed_value"]), pd.to_numeric(data["open_value"])]
    ).dropna()
    lower, upper = float(values.min()), float(values.max())
    axis.plot([lower, upper], [lower, upper], color="#888888", linestyle="--", linewidth=0.7)
    axis.set_xlabel("Closed-model mean")
    axis.set_ylabel("Open-model mean")
    axis.legend(frameon=False, fontsize=5.3, ncol=2)
    panel_title(axis, "a", "Behavioural correspondence")


def _readout(axis, frame: pd.DataFrame) -> None:
    data = frame[frame["panel"] == "b"].copy()
    for theory, group in data.groupby("theory", sort=False):
        group = group.sort_values("decoder_block")
        axis.plot(
            pd.to_numeric(group["decoder_block"]),
            pd.to_numeric(group["estimate"]),
            color=color_for(theory),
            linewidth=1.0,
            marker="o",
            markersize=2.8,
            label=theory,
        )
        selected = group[group["selected"].astype(str).str.lower() == "true"]
        axis.scatter(
            pd.to_numeric(selected["decoder_block"]),
            pd.to_numeric(selected["estimate"]),
            s=34,
            facecolor="none",
            edgecolor=color_for(theory),
            linewidth=1.0,
        )
    axis.axhline(0.90, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_ylim(0.45, 1.03)
    axis.set_xlabel("Decoder block")
    axis.set_ylabel("Confirmation AUC")
    axis.legend(frameon=False, fontsize=5.3)
    panel_title(axis, "b", "Linear readout across blocks")


def _activation_edits(axis, frame: pd.DataFrame) -> None:
    data = frame[frame["panel"] == "c"].copy().reset_index(drop=True)
    data["label"] = data[["theory", "metric"]].fillna("").agg(" | ".join, axis=1).str.strip(" |")
    for index, row in data.iterrows():
        draw_interval(
            axis,
            float(row["estimate"]),
            index,
            float(row["ci_low"]),
            float(row["ci_high"]),
            color="#44546A",
            markerfacecolor="white",
        )
    axis.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
    axis.set_yticks(range(len(data)), [short_label(value, 32) for value in data["label"]])
    axis.set_xlabel("Activation-edit estimate with 95% interval")
    panel_title(axis, "c", "Numerical-relation activation edits")


def _state_replacement(axis, frame: pd.DataFrame) -> None:
    axis.set_axis_off()
    data = frame[frame["panel"] == "d"].copy()
    grid = data[data["theme"].notna() & data["dimension"].notna()]
    themes = list(dict.fromkeys(grid["theme"].tolist()))
    dimensions = list(dict.fromkeys(grid["dimension"].tolist()))
    conditions = ["Clean", "Target patch", "Random patch"]
    finite = pd.to_numeric(grid["estimate"], errors="coerce").dropna()
    limit = max(abs(float(finite.min())), abs(float(finite.max())))
    image = None
    for index, condition in enumerate(conditions):
        inset = axis.inset_axes([index * 0.255, 0.12, 0.225, 0.80])
        subset = grid[grid["condition"] == condition]
        matrix = np.full((len(themes), len(dimensions)), np.nan)
        for theme_index, theme in enumerate(themes):
            for dimension_index, dimension in enumerate(dimensions):
                match = subset[(subset["theme"] == theme) & (subset["dimension"] == dimension)]
                if not match.empty:
                    matrix[theme_index, dimension_index] = float(match.iloc[0]["estimate"])
        image = inset.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        inset.set_title(condition, fontsize=6.4)
        inset.set_xticks(range(len(dimensions)), [item[:3] for item in dimensions], rotation=55, ha="right")
        inset.set_yticks(
            range(len(themes)),
            [item.replace("_elements", "").title() for item in themes] if index == 0 else [],
        )
    if image is not None:
        color_axis = axis.inset_axes([0.75, 0.60, 0.012, 0.28])
        colorbar = plt.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=4.8)
    summary = data[data["theme"].isna() & data["estimate"].notna()].reset_index(drop=True)
    forest = axis.inset_axes([0.80, 0.12, 0.19, 0.78])
    for index, row in summary.iterrows():
        draw_interval(
            forest,
            float(row["estimate"]),
            index,
            float(row["ci_low"]),
            float(row["ci_high"]),
            color="#7656A3",
            markerfacecolor="white",
        )
    forest.axvline(0, color="#888888", linestyle="--", linewidth=0.7)
    forest.set_yticks(range(len(summary)), [short_label(value, 14) for value in summary["condition"]])
    forest.yaxis.tick_right()
    forest.tick_params(axis="y", labelsize=5.0, pad=1)
    forest.set_xlabel("Estimate")
    panel_title(axis, "d", "Perception state replacement")


def render(output_dir: Path) -> dict:
    apply_style()
    frame = pd.read_csv(SOURCE)
    figure = plt.figure(figsize=(9.0, 6.1))
    grid = figure.add_gridspec(2, 2, left=0.08, right=0.98, top=0.94, bottom=0.09, wspace=0.42, hspace=0.47)
    _behaviour(figure.add_subplot(grid[0, 0]), frame)
    _readout(figure.add_subplot(grid[0, 1]), frame)
    _activation_edits(figure.add_subplot(grid[1, 0]), frame)
    _state_replacement(figure.add_subplot(grid[1, 1]), frame)
    return save_bundle(figure, output_dir / "figure5_open_model_analysis_public_rerender")
