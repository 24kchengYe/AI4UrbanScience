from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"
FIGURE_SOURCE_ROOT = DATA_ROOT / "figure_sources"
PUBLICATION_DATA_ROOT = DATA_ROOT / "publication"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
REFERENCE_OUTPUT_ROOT = REPOSITORY_ROOT / "reference_outputs"


def repository_path(relative: str | Path) -> Path:
    """Resolve a repository-relative path and reject traversal outside the tree."""

    resolved = (REPOSITORY_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {relative}") from error
    return resolved

