from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..source_data import SOURCES
from .style import apply_style, color_for, draw_interval, panel_title, save_bundle


LABEL_COLUMNS = {
    2: ("component", "metric", "requested_n", "changed_input"),
    3: ("source", "variable", "metric", "empirical_definition", "changed_input"),
    4: ("source", "metric"),
    5: ("model", "condition", "metric", "rings", "grid"),
    6: ("source", "feature", "variant", "metric", "component"),
    7: ("factor", "control", "metric", "requested_n"),
    8: ("dimension", "component", "metric"),
    9: ("encoder", "theme", "dimension", "condition", "metric"),
    11: ("theory", "stage", "metric", "hidden_state_index", "pass_flag"),
    12: ("condition", "theme", "dimension"),
}


def panel_row_labels(
    data: pd.DataFrame,
    number: int,
    *,
    exclude_columns: tuple[str, ...] = (),
) -> list[str]:
    """Construct complete, panel-specific labels without dropping dimensions."""

    columns = [
        column
        for column in LABEL_COLUMNS[number]
        if column in data and column not in exclude_columns
    ]
    labels: list[str] = []
    for _, row in data.iterrows():
        parts = []
        for column in columns:
            if pd.isna(row[column]) or not str(row[column]).strip():
                continue
            value = str(row[column]).strip().replace("_", " ")
            parts.append(value)
        labels.append(" | ".join(parts) or "estimate")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Figure S{number:02d} panel labels are not unique")
    return labels


def _table_figure(frame: pd.DataFrame, number: int, output_dir: Path) -> dict:
    apply_style()
    figure, axis = plt.subplots(figsize=(8.7, 2.8))
    axis.set_axis_off()
    display = frame.drop(columns=["panel"], errors="ignore").fillna("")
    table = axis.table(
        cellText=display.values,
        colLabels=[str(item).replace("_", " ").title() for item in display.columns],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1.0, 1.45)
    axis.set_title("Case definitions and released inputs", loc="left", fontsize=9.0, pad=10)
    return save_bundle(figure, output_dir / f"figureS{number:02d}_source_data_public_rerender")


def _correspondence_figure(frame: pd.DataFrame, number: int, output_dir: Path) -> dict:
    apply_style()
    panels = list(dict.fromkeys(frame["panel"].tolist()))
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    plotted_rows = 0
    for axis, panel, letter in zip(axes.flat, panels, "abcd"):
        data = frame[frame["panel"] == panel]
        paired_values = data[["closed_mean", "open_mean"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if paired_values.isna().any(axis=None):
            missing = int(paired_values.isna().any(axis=1).sum())
            raise ValueError(
                f"Figure S{number:02d} panel {panel} has {missing} unplottable released rows"
            )
        for theory, group in data.groupby("theory", sort=False):
            axis.scatter(
                pd.to_numeric(group["closed_mean"]),
                pd.to_numeric(group["open_mean"]),
                s=14,
                alpha=0.78,
                color=color_for(str(theory).title()),
            )
            plotted_rows += len(group)
        values = pd.concat([pd.to_numeric(data["closed_mean"]), pd.to_numeric(data["open_mean"])]).dropna()
        if not values.empty:
            lower, upper = float(values.min()), float(values.max())
            axis.plot([lower, upper], [lower, upper], color="#888888", linestyle="--", linewidth=0.7)
        correlation = pd.to_numeric(data["pearson_r"], errors="coerce").dropna()
        title = str(data["theory"].iloc[0]).replace("_", " ").title()
        if not correlation.empty:
            title += f" (r={correlation.iloc[0]:.2f})"
        axis.set_xlabel("Closed-model mean")
        axis.set_ylabel("Open-model mean")
        panel_title(axis, letter, title)
    if plotted_rows != len(frame):
        raise ValueError(
            f"Figure S{number:02d} rendered {plotted_rows} of {len(frame)} released rows"
        )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.09, wspace=0.36, hspace=0.42)
    return save_bundle(figure, output_dir / f"figureS{number:02d}_source_data_public_rerender")


def _estimate_figure(frame: pd.DataFrame, number: int, output_dir: Path) -> dict:
    apply_style()
    panels = list(dict.fromkeys(frame["panel"].tolist()))
    panel_heights = [
        max(2.8, 1.5 + 0.185 * len(frame[frame["panel"] == panel]))
        for panel in panels
    ]
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.0, sum(panel_heights)),
        squeeze=False,
        gridspec_kw={"height_ratios": panel_heights},
    )
    plotted_rows = 0
    for axis, panel, letter in zip(axes.flat, panels, "abcdefghijklmnopqrstuvwxyz"):
        data = frame[frame["panel"] == panel].copy().reset_index(drop=True)
        value_column = "pass_flag" if number == 11 and str(panel) == "d" else "estimate"
        if value_column not in data:
            raise ValueError(
                f"Figure S{number:02d} panel {panel} has no supported value column"
            )
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
        if data[value_column].isna().any():
            missing = int(data[value_column].isna().sum())
            raise ValueError(
                f"Figure S{number:02d} panel {panel} has {missing} unplottable released rows"
            )
        labels = panel_row_labels(
            data,
            number,
            exclude_columns=(value_column,) if value_column != "estimate" else (),
        )
        for index, (_, row) in enumerate(data.iterrows()):
            low = pd.to_numeric(pd.Series([row.get("ci_low", np.nan)]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([row.get("ci_high", np.nan)]), errors="coerce").iloc[0]
            draw_interval(
                axis,
                float(row[value_column]),
                index,
                float(low),
                float(high),
                color="#44546A",
                markerfacecolor="white",
            )
        plotted_rows += len(data)
        axis.axvline(0, color="#999999", linestyle="--", linewidth=0.6)
        axis.set_yticks(
            range(len(data)),
            [textwrap.fill(value, width=44) for value in labels],
        )
        axis.invert_yaxis()
        if value_column == "pass_flag":
            axis.set_xlim(-0.08, 1.08)
            axis.set_xticks([0, 1], ["Fail", "Pass"])
            axis.set_xlabel("Released pass criterion")
        else:
            axis.set_xlabel("Released estimate with interval where available")
        panel_title(axis, letter, f"Panel {panel}")
    if plotted_rows != len(frame):
        raise ValueError(
            f"Figure S{number:02d} rendered {plotted_rows} of {len(frame)} released rows"
        )
    figure.subplots_adjust(left=0.36, right=0.98, top=0.96, bottom=0.05, hspace=0.42)
    return save_bundle(figure, output_dir / f"figureS{number:02d}_source_data_public_rerender")


def render(number: int, output_dir: Path) -> dict:
    if number not in range(1, 13):
        raise ValueError("Supplementary figure number must be 1 through 12")
    frame = pd.read_csv(SOURCES[f"FigS{number:02d}_data"])
    if number == 1:
        return _table_figure(frame, number, output_dir)
    if number == 10:
        return _correspondence_figure(frame, number, output_dir)
    return _estimate_figure(frame, number, output_dir)
