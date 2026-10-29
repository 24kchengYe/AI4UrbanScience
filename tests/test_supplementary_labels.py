"""Reader labels must not alter any released estimate or plotted data artist."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ai4us.figures import supplementary
from ai4us.source_data import SOURCES


SOURCE_HASHES = {
    2: "58cd15d4c7a6007e81b06b14dcebefe378ae54ef8dfb5e78c4cbe3745332588f",
    3: "a65d3818b4e345f6b27529b9cdc80843e5e3d0c045a45a3626e66775fc53396b",
    7: "5093f6e4c1f0af6573a2045bd16ee29e0949e2620091a14663a52645b9446eb2",
    12: "9a9ca73b9e2486862437f0a3107208ef138e1b13f157717aba699899475a9fc5",
}


def _frame(number):
    return pd.read_csv(SOURCES[f"FigS{number:02d}_data"])


def _labels(number, panel):
    frame = _frame(number)
    return supplementary.panel_row_labels(frame[frame.panel == panel], number)


def test_corrected_labels_are_specific_to_their_panels():
    component_labels = _labels(2, "a")
    assert sum("City-system organization" in x for x in component_labels) == 2
    assert sum("Numerical range" in x for x in component_labels) == 2
    assert sum("Heterogeneity" in x for x in component_labels) == 2
    for panel in ("b", "c"):
        assert all("requested generic cities" in x for x in _labels(2, panel))
    for number in (2, 3):
        labels = _labels(number, "d")
        assert "Road → GDP" in labels[0]
        assert "GDP → Road" in labels[1]
    for panel in ("a", "b"):
        assert sum("Administrative cities" in x for x in _labels(3, panel)) == 3
        assert "administrative-city reference" in supplementary.PANEL_TITLES[(3, panel)]
    assert any("area > 20 km²" in x for x in _labels(3, "c"))
    assert any("area > 5 km²" in x for x in _labels(3, "c"))
    assert supplementary.PANEL_TITLES[(7, "b")] == (
        "Functional Mix controls (16/20 valid in both requests)"
    )
    assert _labels(12, "b") == [
        "All image states (reduction)",
        "High-change states (reduction)",
        "High-change minus |random-image control|",
        "High-change minus |low-change control|",
        "High-change minus |random-text control|",
    ]


@pytest.mark.parametrize("number", SOURCE_HASHES)
def test_label_mapping_preserves_frozen_sources_and_all_rows(number):
    path = SOURCES[f"FigS{number:02d}_data"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASHES[number]
    frame = _frame(number)
    before = frame.copy(deep=True)
    for _, panel in frame.groupby("panel", sort=False):
        labels = supplementary.panel_row_labels(panel, number)
        assert len(labels) == len(panel) == len(set(labels))
    pd.testing.assert_frame_equal(frame, before)


def _numeric_artists(figure):
    """Capture positions, values, intervals and colours, excluding label text."""
    result = {"size": figure.get_size_inches().tolist(), "axes": []}
    for axis in figure.axes:
        result["axes"].append({
            "position": list(axis.get_position().bounds),
            "limits": [list(axis.get_xlim()), list(axis.get_ylim())],
            "lines": [
                {
                    "x": np.asarray(line.get_xdata(), dtype=float).tolist(),
                    "y": np.asarray(line.get_ydata(), dtype=float).tolist(),
                    "colour": line.get_color(),
                    "markerface": line.get_markerfacecolor(),
                    "marker": line.get_marker(),
                }
                for line in axis.lines
            ],
            "collections": [
                {
                    "paths": [path.vertices.tolist() for path in collection.get_paths()],
                    "facecolours": collection.get_facecolors().tolist(),
                    "edgecolours": collection.get_edgecolors().tolist(),
                }
                for collection in axis.collections
            ],
        })
    return result


@pytest.mark.parametrize("number", SOURCE_HASHES)
def test_display_changes_leave_numeric_artists_unchanged(number, monkeypatch):
    # Capture the figure in memory; no generated files enter the repository.
    monkeypatch.setattr(supplementary, "save_bundle", lambda figure, _: figure)
    frame = _frame(number)
    corrected = supplementary._estimate_figure(frame, number, Path("unused"))
    corrected_artists = _numeric_artists(corrected)
    for axis, (_, panel) in zip(corrected.axes, frame.groupby("panel", sort=False)):
        points = [line for line in axis.lines if line.get_marker() == "o"]
        assert len(points) == len(panel)
        np.testing.assert_allclose(
            [float(line.get_xdata()[0]) for line in points], panel.estimate.to_numpy()
        )
        np.testing.assert_array_equal(
            [line.get_ydata()[0] for line in points], np.arange(len(panel))
        )
        assert all(line.get_color() == "#44546A" for line in points)
        intervals = [
            (index, row)
            for index, (_, row) in enumerate(panel.iterrows())
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high)
        ]
        assert len(axis.collections) == len(intervals)
        for collection, (index, row) in zip(axis.collections, intervals):
            np.testing.assert_allclose(
                collection.get_segments()[0],
                [[row.ci_low, index], [row.ci_high, index]],
            )
    monkeypatch.setattr(
        supplementary, "_display_label",
        lambda number, panel, column, value: str(value).strip().replace("_", " "),
    )
    monkeypatch.setattr(supplementary, "PANEL_TITLES", {})
    plain = supplementary._estimate_figure(frame, number, Path("unused"))
    assert _numeric_artists(plain) == corrected_artists
    plt.close(corrected)
    plt.close(plain)


def test_text_control_contrast_retains_its_negative_estimate_and_interval():
    data = _frame(12)
    row = data[data.condition == "vs text-random"].iloc[0]
    assert row.estimate == pytest.approx(-2.9742328917978154)
    assert row.ci_low == pytest.approx(-3.7404754220919467)
    assert row.ci_high == pytest.approx(-2.2367876612917876)
