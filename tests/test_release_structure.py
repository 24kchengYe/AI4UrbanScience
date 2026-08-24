import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ai4us.figures.supplementary import panel_row_labels
from ai4us.figures.style import color_for


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ID = re.compile(r"ai4us-artifact-sha256-([0-9a-f]{64})")


def _valid_artifact_identity(row: dict[str, str]) -> bool:
    match = ARTIFACT_ID.fullmatch(row["artifact_id"])
    return bool(match and match.group(1) == row["sha256"])


def test_restricted_exact_inputs_are_absent() -> None:
    forbidden = {
        "real_generated_profile_metrics.csv",
        "vitality_five_X52.csv",
        "pedestrian_activity_Y52.csv",
    }
    observed = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert not forbidden.intersection(observed)
    assert not [
        name
        for name in observed
        if name.endswith((".source_curves.csv", ".source_parameters.csv"))
    ]


def test_processed_data_uses_reader_oriented_namespace() -> None:
    directories = {
        path.relative_to(ROOT / "data/processed").as_posix()
        for path in (ROOT / "data/processed").iterdir()
        if path.is_dir()
    }
    assert directories == {"model_outputs"}
    assert len(list((ROOT / "data/processed/model_outputs").glob("*.csv"))) == 32


def test_figure_release_structure_and_complete_diagnostic_labels() -> None:
    assert (ROOT / "reference_outputs/main/figure1_conceptual.pdf").is_file()
    assert not list((ROOT / "src").rglob("*figure1*"))
    assert not (ROOT / "provenance" / ("original_" + "manifests")).exists()
    assert not list(ROOT.rglob("*.zip"))
    figure_index = json.loads((ROOT / "data/figure_sources/manifest.json").read_text(encoding="utf-8"))
    assert figure_index["figure_count"] == 17
    assert len(figure_index["figures"]) == 17
    reader_entries = [entry for entry in figure_index["figures"] if entry["reader_manifest"]]
    assert len(reader_entries) == 16
    for entry in reader_entries:
        assert (ROOT / entry["source_table"]).is_file()
        assert (ROOT / entry["reader_manifest"]).is_file()
        reader = json.loads((ROOT / entry["reader_manifest"]).read_text(encoding="utf-8"))
        assert "historical_" + "lineage" not in reader
        lineage = reader["author_side_lineage"]
        assert set(lineage) == {"artifact_id", "sha256", "availability", "role"}
        assert _valid_artifact_identity(lineage)
        assert lineage["availability"] == "not_bundled"
    with (ROOT / "provenance/dependency_inventory.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        dependency_reader = csv.DictReader(handle)
        assert dependency_reader.fieldnames == [
            "artifact_id",
            "source_status",
            "sha256",
            "disposition",
            "public_replacement",
            "note",
        ]
        dependency_rows = list(dependency_reader)
    with (ROOT / "provenance/artifact_lineage.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        lineage_reader = csv.DictReader(handle)
        assert lineage_reader.fieldnames == [
            "artifact_id",
            "sha256",
            "disposition",
            "public_replacement",
            "note",
        ]
        lineage_rows = list(lineage_reader)
    assert len(dependency_rows) == len({row["artifact_id"] for row in dependency_rows}) == 131
    assert len(lineage_rows) == len({row["artifact_id"] for row in lineage_rows}) == 59
    assert all(_valid_artifact_identity(row) for row in dependency_rows)
    assert all(_valid_artifact_identity(row) for row in lineage_rows)
    assert {row["artifact_id"] for row in lineage_rows}.issubset(
        {row["artifact_id"] for row in dependency_rows}
    )
    parsed_manifest_path = ROOT / "data/release_manifests/parsed_model_outputs_manifest.csv"
    with parsed_manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        parsed_rows = list(csv.DictReader(handle))
    assert len(parsed_rows) == 32
    assert len(list((ROOT / "data/processed/model_outputs").glob("*.csv"))) == 32
    for row in parsed_rows:
        path = ROOT / row["public_file"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert _valid_artifact_identity(row)
    parsed_reference = json.loads(
        (ROOT / "data/release_manifests/parsed_model_outputs_reference.json").read_text(
            encoding="utf-8"
        )
    )
    assert parsed_reference["external_archive"]["availability"] == "not_bundled"
    assert parsed_reference["external_member_manifest"]["availability"] == "not_bundled"
    assert parsed_reference["public_extract"]["csv_count"] == 32
    for number in (3, 11, 12):
        path = ROOT / f"data/figure_sources/supplementary/FigS{number:02d}/FigS{number:02d}.source.csv"
        frame = pd.read_csv(path)
        total_labels = 0
        for _, panel in frame.groupby("panel", sort=False):
            labels = panel_row_labels(panel, number)
            assert len(labels) == len(panel)
            assert len(labels) == len(set(labels))
            total_labels += len(labels)
        assert total_labels == len(frame)
    assert len(pd.read_csv(ROOT / "data/figure_sources/supplementary/FigS12/FigS12.source.csv")) > 12
    assert color_for("Baseline prompt") != color_for("Blueprint prompt")
