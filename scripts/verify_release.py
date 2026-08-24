from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

from ai4us.source_data import validate_source_data


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/publication/source_data.xlsx": "68999352045c5d06fb4d98d9204d140fa915385a8f628957907b17f101e71883",
    "data/publication/supplementary_data_1_prompts_and_inference.xlsx": "cff9a0ae44e52259ae1bcb08856097c7582971d2ddca650920e4c87c981f3a42",
    "data/publication/supplementary_data_2_empirical_provenance.xlsx": "6b7c0f69dd03c64ca92f77deb46d11a2098b51f5a40de964b793291c821ac92c",
    "reference_outputs/main/figure1_conceptual.pdf": "974108e283a0dffe245e970996545c9995155b7e508b145d75d94e02e71f0ef4",
    "reference_outputs/main/figure2_relationships.pdf": "723acc513c236d95a6737a479303fb0d967413eb5f95d5965aa04318c73b52d3",
    "reference_outputs/main/figure3_empirical_variation.pdf": "928d799d1cdd78bba36c74b1a78c0d1124921cac4fa1d4cbb836c2063e10d1c2",
    "reference_outputs/main/figure4_conditional_interventions.pdf": "d960f602733c1efcb42ffd9fa0a5508e4ff0c9859a1fc161aa8c508f0c6464ea",
    "reference_outputs/main/figure5_open_model_analysis.pdf": "41621b063c19bc4890a0724a11616ab393271ec5101fbc90fc5df35e6be71877",
}
EXTERNAL_ARCHIVE_SHA256 = "5b0820e2f0745b1a1ee11d7a627cc6acf62d86e013f901eb784614b1a38e163f"
EXTERNAL_MEMBER_MANIFEST_SHA256 = "5c0ddb5cd57a1b2af22192dc30a3b38ece1903b03d91efc66e0cb6c0d8963613"
FORBIDDEN_BASENAMES = {
    "real_generated_profile_metrics.csv",
    "vitality_five_X52.csv",
    "pedestrian_activity_Y52.csv",
}
FORBIDDEN_SUFFIXES = (".source_curves.csv", ".source_parameters.csv")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".csv",
    ".cff",
    ".example",
    ".gitignore",
}
CONTAINER_TEXT_SUFFIXES = TEXT_SUFFIXES | {".xml", ".rels", ".tsv"}
GENERATED_DIRECTORIES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "outputs",
    "build",
    "dist",
}
ARTIFACT_ID_RE = re.compile(r"ai4us-artifact-sha256-([0-9a-f]{64})")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def repository_files() -> list[Path]:
    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in GENERATED_DIRECTORIES and not name.endswith(".egg-info")
        ]
        paths.extend(Path(directory) / name for name in filenames)
    return paths


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _artifact_identity_is_valid(row: dict[str, str]) -> bool:
    match = ARTIFACT_ID_RE.fullmatch(row.get("artifact_id", ""))
    return bool(match and match.group(1) == row.get("sha256"))


def _replacement_failures(rows: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        if row["disposition"] == "outside_public_code_repository":
            continue
        for replacement in filter(None, row["public_replacement"].split(";")):
            if not (ROOT / replacement).is_file():
                failures.append(f"{row['artifact_id']} -> {replacement}")
    return failures


def _ordinary_text(path: Path) -> str | None:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".pdf":
        payload = re.sub(
            rb"stream(?:\r\n|\r|\n).*?endstream",
            b"",
            path.read_bytes(),
            flags=re.DOTALL,
        )
        return payload.decode("latin-1", errors="replace")
    return None


def _public_surfaces(files: list[Path]):
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        yield relative, relative
        text = _ordinary_text(path)
        if text is not None:
            yield relative, text
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    location = f"{relative}::{member.filename}"
                    yield location, member.filename
                    suffix = Path(member.filename).suffix.lower()
                    if not member.is_dir() and suffix in CONTAINER_TEXT_SUFFIXES:
                        yield location, archive.read(member).decode(
                            "utf-8-sig", errors="replace"
                        )


def _scan_surface_patterns(files: list[Path]) -> tuple[list[dict], list[dict]]:
    authoring_interface = "".join(chr(value) for value in (67, 111, 100, 101, 120))
    visual_interface = "image" + "_" + "gen"
    workspace_marker = "Research" + "_20250121"
    class_a = [
        ("authoring-interface", re.compile(re.escape(authoring_interface), re.I)),
        ("visual-authoring-interface", re.compile(re.escape(visual_interface), re.I)),
        ("workspace-marker", re.compile(re.escape(workspace_marker), re.I)),
        (
            "windows-absolute",
            re.compile(r"[A-Za-z]:[\\/](?:Users|Research|Data|Temp)[\\/]", re.I),
        ),
        ("posix-private", re.compile(r"(?<!:)/(?:Users|home|tmp)/", re.I)),
        (
            "network-private",
            re.compile(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+", re.I),
        ),
        ("file-uri", re.compile("file" + "://", re.I)),
        (
            "temporary-location",
            re.compile(
                "App" + r"Data[\\/]Local[\\/]Temp|" + "%" + "TEMP" + "%",
                re.I,
            ),
        ),
        (
            "credential-value",
            re.compile(
                r"(?i)(api[_-]?key|access[_-]?token|secret|password)"
                r"\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{16,}"
            ),
        ),
    ]
    class_b = [
        ("stage-tree", re.compile("revi" + r"sion[\\/]", re.I)),
        (
            "numbered-author-stage",
            re.compile("(?:work" + "station_)?form" + r"al[_-]?v[0-9]+", re.I),
        ),
        ("machine-label", re.compile("work" + "station", re.I)),
        (
            "release-label",
            re.compile(
                "(?:main_"
                + "v[0-9]+|supplementary_"
                + "v[0-9]+|main-ready-"
                + "v[0-9]+|source_code_"
                + "formal|current_"
                + "v[0-9]+)",
                re.I,
            ),
        ),
        ("old-path-column", re.compile("(?:prior_" + "path|legacy_" + "path)", re.I)),
        ("old-snapshot-tree", re.compile("original_" + "manifests", re.I)),
        ("old-lineage-key", re.compile("historical_" + "lineage", re.I)),
        ("old-mapping-name", re.compile("legacy_" + "mapping", re.I)),
        ("old-data-label", re.compile("parsed_model_outputs_" + "v12", re.I)),
        (
            "old-archive-name",
            re.compile("supplementary_data_3_" + "parsed_model_outputs", re.I),
        ),
    ]
    allowed_email = "ylong" + "@" + "tsinghua.edu.cn"
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    a_hits: list[dict] = []
    b_hits: list[dict] = []
    for location, surface in _public_surfaces(files):
        for name, pattern in class_a:
            if pattern.search(surface):
                a_hits.append({"surface": location, "class": name})
        unexpected_emails = {
            value.casefold()
            for value in email_re.findall(surface)
            if value.casefold() != allowed_email.casefold()
        }
        if unexpected_emails:
            a_hits.append({"surface": location, "class": "unexpected-account-identifier"})
        for name, pattern in class_b:
            if pattern.search(surface):
                b_hits.append({"surface": location, "class": name})
    return a_hits, b_hits


def _append_check(checks: list[dict], name: str, failures: list, **evidence) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if not failures else "FAIL",
            **evidence,
            "failures": failures,
        }
    )


def verify() -> dict:
    checks: list[dict] = []
    files = repository_files()

    authority_failures = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        observed = digest(path) if path.is_file() else None
        if observed != expected:
            authority_failures.append(
                {"path": relative, "expected": expected, "observed": observed}
            )
    _append_check(checks, "frozen_authority_hashes", authority_failures)

    source_report = validate_source_data()
    source_failures = [] if source_report["status"] == "PASS" else [source_report]
    _append_check(
        checks,
        "source_data_cell_matrix",
        source_failures,
        equal_sheets=source_report["cell_matrix_equal_count"],
        sheet_count=source_report["sheet_count"],
        formula_cells=source_report["formula_cell_count"],
    )

    reader_failures = []
    figure_index_path = ROOT / "data/figure_sources/manifest.json"
    if not figure_index_path.is_file():
        reader_failures.append("figure index missing")
    else:
        figure_index = _load_json(figure_index_path)
        entries = figure_index.get("figures", [])
        if figure_index.get("figure_count") != 17 or len(entries) != 17:
            reader_failures.append("figure index must contain 17 entries")
        for entry in entries:
            if entry.get("figure") == "1":
                reference = ROOT / str(entry.get("reference_output"))
                if entry.get("renderer") is not None or not reference.is_file():
                    reader_failures.append("Figure 1 must be reference-only")
                continue
            manifest_path = ROOT / str(entry.get("reader_manifest"))
            source_path = ROOT / str(entry.get("source_table"))
            if not manifest_path.is_file() or not source_path.is_file():
                reader_failures.append(str(entry.get("figure")))
                continue
            reader = _load_json(manifest_path)
            if reader.get("source_table", {}).get("sha256") != digest(source_path):
                reader_failures.append(f"{entry.get('figure')}: source hash")
            lineage = reader.get("author_side_lineage", {})
            if (
                not _artifact_identity_is_valid(lineage)
                or lineage.get("availability") != "not_bundled"
            ):
                reader_failures.append(f"{entry.get('figure')}: lineage identity")
            module_path = ROOT / "src" / Path(*str(entry.get("renderer")).split("."))
            if not module_path.with_suffix(".py").is_file():
                reader_failures.append(f"{entry.get('figure')}: renderer module")
            reference_value = entry.get("reference_output")
            if reference_value is not None and not (ROOT / str(reference_value)).is_file():
                reader_failures.append(f"{entry.get('figure')}: reference output")
        supplementary_path = ROOT / "data/figure_sources/supplementary/manifest.json"
        if not supplementary_path.is_file():
            reader_failures.append("supplementary index missing")
        else:
            supplementary = _load_json(supplementary_path)
            if supplementary.get("figure_count") != 12:
                reader_failures.append("supplementary index count")
            if not _artifact_identity_is_valid(
                supplementary.get("author_side_lineage", {})
            ):
                reader_failures.append("supplementary index lineage")
        if not _artifact_identity_is_valid(figure_index.get("author_side_lineage", {})):
            reader_failures.append("figure index lineage")
    _append_check(checks, "current_reader_manifests", reader_failures)

    lineage_path = ROOT / "provenance/artifact_lineage.csv"
    lineage_failures = []
    if not lineage_path.is_file():
        lineage_rows = []
        lineage_failures.append("artifact lineage table missing")
    else:
        lineage_fields, lineage_rows = _read_csv(lineage_path)
        expected_fields = [
            "artifact_id",
            "sha256",
            "disposition",
            "public_replacement",
            "note",
        ]
        if lineage_fields != expected_fields:
            lineage_failures.append("artifact lineage schema")
        if len(lineage_rows) != 59 or len(
            {row["artifact_id"] for row in lineage_rows}
        ) != 59:
            lineage_failures.append("artifact lineage cardinality")
        if not all(_artifact_identity_is_valid(row) for row in lineage_rows):
            lineage_failures.append("artifact lineage identity")
        lineage_failures.extend(_replacement_failures(lineage_rows))
    _append_check(
        checks,
        "public_artifact_lineage",
        lineage_failures,
        rows=len(lineage_rows),
    )

    dependency_path = ROOT / "provenance/dependency_inventory.csv"
    dependency_failures = []
    if not dependency_path.is_file():
        dependency_rows = []
        dependency_failures.append("dependency inventory missing")
    else:
        dependency_fields, dependency_rows = _read_csv(dependency_path)
        expected_fields = [
            "artifact_id",
            "source_status",
            "sha256",
            "disposition",
            "public_replacement",
            "note",
        ]
        if dependency_fields != expected_fields:
            dependency_failures.append("dependency inventory schema")
        if len(dependency_rows) != 131 or len(
            {row["artifact_id"] for row in dependency_rows}
        ) != 131:
            dependency_failures.append("dependency inventory cardinality")
        if not all(_artifact_identity_is_valid(row) for row in dependency_rows):
            dependency_failures.append("dependency inventory identity")
        if not {row["artifact_id"] for row in lineage_rows}.issubset(
            {row["artifact_id"] for row in dependency_rows}
        ):
            dependency_failures.append("lineage identities are not a dependency subset")
        dependency_failures.extend(_replacement_failures(dependency_rows))
    _append_check(
        checks,
        "dependency_inventory",
        dependency_failures,
        rows=len(dependency_rows),
    )

    parsed_failures = []
    parsed_reference_path = (
        ROOT / "data/release_manifests/parsed_model_outputs_reference.json"
    )
    parsed_manifest_path = (
        ROOT / "data/release_manifests/parsed_model_outputs_manifest.csv"
    )
    if not parsed_reference_path.is_file() or not parsed_manifest_path.is_file():
        parsed_rows = []
        parsed_failures.append("parsed-output reader files missing")
    else:
        reference = _load_json(parsed_reference_path)
        external_archive = reference.get("external_archive", {})
        external_members = reference.get("external_member_manifest", {})
        if (
            external_archive.get("sha256") != EXTERNAL_ARCHIVE_SHA256
            or external_archive.get("availability") != "not_bundled"
            or not _artifact_identity_is_valid(external_archive)
        ):
            parsed_failures.append("external archive identity")
        if (
            external_members.get("sha256") != EXTERNAL_MEMBER_MANIFEST_SHA256
            or external_members.get("records") != 54
            or external_members.get("availability") != "not_bundled"
            or not _artifact_identity_is_valid(external_members)
        ):
            parsed_failures.append("external member-ledger identity")
        public_extract = reference.get("public_extract", {})
        if public_extract.get("csv_count") != 32:
            parsed_failures.append("public extract count")
        manifest_binding = public_extract.get("manifest", {})
        if manifest_binding.get("sha256") != digest(parsed_manifest_path):
            parsed_failures.append("public extract manifest hash")
        _, parsed_rows = _read_csv(parsed_manifest_path)
        observed_paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "data/processed/model_outputs").glob("*.csv")
        }
        declared_paths = {row["public_file"] for row in parsed_rows}
        if len(parsed_rows) != 32 or observed_paths != declared_paths:
            parsed_failures.append("public extract file set")
        for row in parsed_rows:
            path = ROOT / row["public_file"]
            if (
                not path.is_file()
                or digest(path) != row["sha256"]
                or path.stat().st_size != int(row["bytes"])
                or not _artifact_identity_is_valid(row)
            ):
                parsed_failures.append(row["artifact_id"])
    if any(path.suffix.lower() == ".zip" for path in files):
        parsed_failures.append("public ZIP container present")
    _append_check(
        checks,
        "parsed_model_output_extracts",
        parsed_failures,
        rows=len(parsed_rows),
    )

    restricted_hashes = {
        row["sha256"]
        for row in dependency_rows
        if row["disposition"] == "exact_private_table_excluded"
    }
    restricted_files = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.name in FORBIDDEN_BASENAMES
        or path.name.endswith(FORBIDDEN_SUFFIXES)
        or digest(path) in restricted_hashes
    ]
    if len(restricted_hashes) != 5:
        restricted_files.append("restricted identity count is not five")
    _append_check(checks, "restricted_exact_inputs_absent", restricted_files)

    bad_public_paths = []
    path_patterns = [
        re.compile("form" + r"al[_-]?v\d+", re.I),
        re.compile("work" + "station", re.I),
        re.compile(r"(^|[/\\])P\d[-_]", re.I),
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if any(pattern.search(relative) for pattern in path_patterns):
            bad_public_paths.append(relative)
    _append_check(checks, "reader_path_names", bad_public_paths)

    env_files = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.name.startswith(".env") and path.name != ".env.example"
    ]
    generated_residue = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.name in GENERATED_DIRECTORIES
        or path.name == ".venv"
        or path.name.endswith(".egg-info")
        or path.suffix in {".pyc", ".pyo"}
    ]
    _append_check(
        checks,
        "environment_and_generated_files",
        [*env_files, *generated_residue],
    )

    class_a_hits, class_b_hits = _scan_surface_patterns(files)
    _append_check(checks, "strict_private_surface_scan", class_a_hits)
    _append_check(checks, "strict_public_naming_scan", class_b_hits)

    manifest_path = ROOT / "manifest/files.sha256"
    manifest_failures = []
    if not manifest_path.is_file():
        manifest_failures.append("checksum list missing")
    else:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file() or digest(path) != expected:
                manifest_failures.append(relative)
    _append_check(checks, "manifest_checksums", manifest_failures)

    status = (
        "PASS"
        if len(checks) == 12 and all(check["status"] == "PASS" for check in checks)
        else "FAIL"
    )
    return {
        "schema_version": "1.0",
        "status": status,
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    report = verify()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
