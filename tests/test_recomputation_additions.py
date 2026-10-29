from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/controlled_scaling"
CUE_DATA = ROOT / "data/processed/single_cue"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scaling = load_module("controlled_scaling_example", EXAMPLE / "run_example.py")
single_cue = load_module("single_cue_reanalysis", ROOT / "scripts/recompute_single_cue.py")


@pytest.fixture
def scaling_inputs():
    contract = json.loads((EXAMPLE / "contract.json").read_text(encoding="utf-8"))
    with (EXAMPLE / "data/urban_scaling_intervention_outputs.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return rows, contract


@pytest.fixture
def cue_inputs():
    return [
        single_cue.csv_rows(CUE_DATA / "perception_single_cue_scores.csv"),
        single_cue.csv_rows(CUE_DATA / "perception_single_cue_scene_estimands.csv"),
        single_cue.csv_rows(CUE_DATA / "perception_single_cue_summary.csv"),
        json.loads((CUE_DATA / "single_cue_analysis_config.json").read_text(encoding="utf-8")),
        single_cue.csv_rows(CUE_DATA / "figure_s9_source.csv"),
    ]


def test_scaling_point_estimates(scaling_inputs):
    units, summary = scaling.analyze(*scaling_inputs)
    assert len(units) == 100
    assert summary["selected_rows"] == 500
    assert summary["other_released_rows_not_analyzed_here"] == 1600
    assert summary["settings_gdp_greater_than_road"] == 99
    assert summary["mean_road_exponent"] == pytest.approx(0.6815397449294656, abs=1e-12)
    assert summary["mean_gdp_exponent"] == pytest.approx(1.0338220034025647, abs=1e-12)


@pytest.mark.parametrize("change", ["missing", "duplicate", "invalid", "nonpositive", "unknown"])
def test_scaling_rejects_changed_design(scaling_inputs, change):
    rows, contract = deepcopy(scaling_inputs)
    if change == "missing":
        rows.pop(0)
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "invalid":
        rows[0]["request_status"] = "empty"
    elif change == "nonpositive":
        rows[0]["road_length_km"] = "0"
    else:
        rows[0]["setting_id"] = "unknown_setting"
    with pytest.raises(ValueError):
        scaling.analyze(rows, contract)


def test_scaling_power_law_and_constant_input():
    assert scaling.slope([1, 2, 4], [3, 12, 48]) == pytest.approx(2)
    with pytest.raises(ValueError):
        scaling.slope([1, 1], [2, 3])


def test_scaling_outputs_and_no_overwrite(tmp_path):
    output = tmp_path / "results"
    report = scaling.run(output)
    assert report["new_model_calls"] == 0
    assert report["matches_original_figure4_point_estimates"] is True
    assert json.loads((output / "summary.json").read_text())["analyzed_settings"] == 100
    with (output / "setting_estimates.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 100
    original = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        scaling.run(output)
    assert (output / "summary.json").read_bytes() == original


def test_scaling_rejects_repository_output():
    with pytest.raises(ValueError, match="outside the repository"):
        scaling.run(EXAMPLE / "unused_test_output")
    assert not (EXAMPLE / "unused_test_output").exists()


def test_scaling_cli_is_independent_of_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", str(EXAMPLE / "run_example.py"), "--output-dir", str(tmp_path / "cli")],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["selected_rows"] == 500


def test_single_cue_effects_and_intervals():
    report = single_cue.recompute(CUE_DATA)
    assert report["status"] == "pass"
    assert report["independent_scenes"] == 12
    assert report["scored_images"] == 60
    assert report["scene_effects_checked"] == 60
    assert report["bootstrap_resamples"] == 10000
    assert report["bootstrap_seed"] == 20260728
    assert report["matches_released_summary_and_FigS9d"] is True
    expected = np.array([
        [1.2083333333333333, 0.6656250000000009, 1.8333333333333333],
        [-1.525, -2.3583333333333334, -0.6666666666666666],
        [1.0416666666666667, 0.5416666666666666, 1.5416666666666667],
        [0.9583333333333334, 0.75, 1.1666666666666667],
        [0.9416666666666668, 0.675, 1.25],
    ])
    observed = [[r[k] for k in ("estimate", "ci95_low", "ci95_high")] for r in report["effects"]]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("change", ["missing", "duplicate", "range", "unknown_scene", "unknown_condition"])
def test_single_cue_rejects_changed_design(cue_inputs, change):
    args = deepcopy(cue_inputs)
    if change == "missing":
        args[0].pop()
    elif change == "duplicate":
        args[0][0] = args[0][1]
    elif change == "range":
        args[0][0]["Beautiful"] = "11"
    elif change == "unknown_scene":
        old = args[0][0]["scene_id"]
        for row in args[0]:
            if row["scene_id"] == old:
                row["scene_id"] = "unknown_scene"
    else:
        args[0][0]["condition"] = "unknown_condition"
    with pytest.raises(ValueError):
        single_cue.recompute_rows(*args)


def test_single_cue_detects_input_corruption(tmp_path):
    copied = tmp_path / "cue_data"
    shutil.copytree(CUE_DATA, copied)
    path = copied / "perception_single_cue_scores.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        single_cue.recompute(copied)


def test_single_cue_cli_does_not_write_files(tmp_path):
    before = {p.relative_to(CUE_DATA): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in CUE_DATA.rglob("*") if p.is_file()}
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/recompute_single_cue.py")],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["status"] == "pass"
    assert not list(tmp_path.iterdir())
    after = {p.relative_to(CUE_DATA): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in CUE_DATA.rglob("*") if p.is_file()}
    assert before == after


def test_added_inputs_match_source_identities():
    record = json.loads((ROOT / "provenance/recomputation_additions_20261029.json").read_text())
    copied = [row for row in record["artifacts"] if row["byte_identical"]]
    assert len(copied) == 11
    for row in copied:
        assert hashlib.sha256((ROOT / row["public_path"]).read_bytes()).hexdigest() == row["source_sha256"]


def test_new_sources_are_covered_by_input_manifests():
    for folder in (EXAMPLE, CUE_DATA):
        manifest = json.loads((folder / "input_manifest.json").read_text())
        for row in manifest["inputs"]:
            assert hashlib.sha256((folder / row["path"]).read_bytes()).hexdigest() == row["sha256"]
