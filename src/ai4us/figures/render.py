from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..paths import REPOSITORY_ROOT
from ..source_data import SOURCES, sha256
from . import conditional_interventions, empirical_variation, open_model_analysis, relationships, supplementary


AVAILABLE = ("2", "3", "4", "5", *(f"S{number:02d}" for number in range(1, 13)))


def _csv_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def _parse_figures(value: str) -> list[str]:
    identifiers = [item.strip().upper() for item in value.split(",") if item.strip()]
    normalized = [f"S{int(item[1:]):02d}" if item.startswith("S") else str(int(item)) for item in identifiers]
    invalid = [item for item in normalized if item not in AVAILABLE]
    if invalid:
        raise ValueError(f"Unknown figure identifiers: {', '.join(invalid)}")
    return list(dict.fromkeys(normalized))


def render_selected(identifiers: list[str], output_root: Path) -> dict:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}. Choose a new directory to preserve prior outputs."
        )
    output_root.mkdir(parents=True)
    records: list[dict] = []
    for identifier in identifiers:
        figure_dir = output_root / f"figure{identifier}"
        if identifier == "2":
            outputs = relationships.render(figure_dir)
            source_key = "Fig02_data"
        elif identifier == "3":
            outputs = empirical_variation.render(figure_dir)
            source_key = "Fig03_data"
        elif identifier == "4":
            outputs = conditional_interventions.render(figure_dir)
            source_key = "Fig04_data"
        elif identifier == "5":
            outputs = open_model_analysis.render(figure_dir)
            source_key = "Fig05_data"
        else:
            number = int(identifier[1:])
            outputs = supplementary.render(number, figure_dir)
            source_key = f"FigS{number:02d}_data"
        source = SOURCES[source_key]
        record = {
            "figure": identifier,
            "status": "PASS",
            "source": source.relative_to(REPOSITORY_ROOT).as_posix(),
            "source_sha256": sha256(source),
            "source_data_rows": _csv_data_rows(source),
            "outputs": outputs,
        }
        if identifier.startswith("S"):
            record["diagnostic_selection"] = {
                "mode": "all_released_rows",
                "selected_rows": record["source_data_rows"],
                "omitted_rows": 0,
            }
        records.append(record)
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "render_type": "public_source_table_rerender",
        "figure_count": len(records),
        "records": records,
    }
    manifest_path = output_root / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Render figures from released figure-source tables.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--figures", help="Comma-separated identifiers, for example 2,4,5,S10")
    choice.add_argument("--all", action="store_true", help="Render Figures 2–5 and S1–S12")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    identifiers = list(AVAILABLE) if args.all else _parse_figures(args.figures)
    manifest = render_selected(identifiers, args.output_dir)
    print(json.dumps({"status": "PASS", "figure_count": manifest["figure_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
