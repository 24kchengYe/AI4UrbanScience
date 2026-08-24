from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifest"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "outputs", "build", "dist"}
EXCLUDED_FILES = {
    "manifest/files.sha256",
    "manifest/release_manifest.json",
    "manifest/verification_report.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included_files() -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_PARTS
            and name != ".venv"
            and not name.endswith(".egg-info")
        ]
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(ROOT)
            if (
                relative.as_posix() == ".git"
                or relative.as_posix() in EXCLUDED_FILES
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def payloads() -> tuple[str, str, dict]:
    records = []
    for path in included_files():
        relative = path.relative_to(ROOT).as_posix()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    checksum_text = "".join(f"{row['sha256']}  {row['path']}\n" for row in records)
    checksum_hash = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "algorithm": "sha256",
        "scope": "tracked release files excluding generated outputs and self-referential manifests",
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "files_sha256": checksum_hash,
        "files": records,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    return checksum_text, manifest_text, manifest


def build() -> dict:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    checksum_text, manifest_text, manifest = payloads()
    checksum_path = MANIFEST_DIR / "files.sha256"
    checksum_path.write_text(checksum_text, encoding="utf-8", newline="\n")
    (MANIFEST_DIR / "release_manifest.json").write_text(
        manifest_text, encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or check the deterministic repository manifest."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed manifest bytes differ from a fresh calculation.",
    )
    args = parser.parse_args()
    if args.check:
        checksum_text, manifest_text, _ = payloads()
        expected = {
            MANIFEST_DIR / "files.sha256": checksum_text,
            MANIFEST_DIR / "release_manifest.json": manifest_text,
        }
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        print(
            json.dumps(
                {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches},
                indent=2,
            )
        )
        return 0 if not mismatches else 1
    manifest = build()
    print(json.dumps({key: manifest[key] for key in ("file_count", "total_bytes", "files_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
