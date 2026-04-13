"""IO helpers for reading and writing generated datasets.

Each experiment script saves its replicates to an Excel or CSV file in a
standard layout (see :class:`ai4us.config.Paths`). The helpers below handle
the tedious parsing/cleanup steps that were previously duplicated across
every experiment script in the codebase.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def parse_delimited(
    response: str,
    columns: Iterable[str],
    *,
    separator: str = ",",
    skip_empty: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """Parse a plain-text LLM response into a :class:`pandas.DataFrame`.

    The model is prompted to return one record per line with fields separated
    by ``separator``. This helper cleans up the common failure modes:

    * trailing/leading whitespace
    * empty lines
    * markdown code fences (triple backticks with optional language)
    * extra explanatory lines that do not contain the separator
    * more fields than expected (extra fields are dropped rather than the row)

    Parameters
    ----------
    response : str
        The raw text returned by the LLM.
    columns : iterable of str
        Expected column names (determines the target column count).
    separator : str, default ``","``
        Field separator.
    skip_empty : bool, default ``True``
        Drop lines that are completely empty.
    strict : bool, default ``False``
        If ``True``, raise a ``ValueError`` when any line has the wrong
        number of fields. If ``False`` (default) such lines are dropped with
        a warning.

    Returns
    -------
    pandas.DataFrame
        One row per successfully-parsed line. Columns have dtype ``object``;
        cast them explicitly if you need numeric types.
    """
    col_list = list(columns)
    n_cols = len(col_list)

    rows: list[list[str]] = []
    for lineno, raw in enumerate(response.splitlines(), start=1):
        line = raw.strip()
        if not line and skip_empty:
            continue
        # drop markdown fences
        if line.startswith("```") or line.endswith("```"):
            continue
        if separator not in line:
            # skip header lines, prose, etc.
            continue
        parts = [p.strip() for p in line.split(separator)]
        if len(parts) < n_cols:
            if strict:
                raise ValueError(
                    f"Line {lineno} has {len(parts)} fields, expected {n_cols}: {raw!r}"
                )
            log.warning("skipping short line %d: %r", lineno, raw)
            continue
        if len(parts) > n_cols:
            # keep the first n_cols fields — often the model adds trailing commas
            parts = parts[:n_cols]
        rows.append(parts)

    return pd.DataFrame(rows, columns=col_list)


def parse_json_map(response: str) -> dict:
    """Parse a JSON object out of an LLM response.

    Strips markdown fences and leading/trailing prose and returns the first
    valid JSON object found in the text. Raises :class:`ValueError` if no
    valid JSON could be extracted.
    """
    text = response.strip()
    # strip triple-backtick fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to isolate the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from response: {e}") from e
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_replicate(df: pd.DataFrame, directory: Path, run_id: int, *,
                   prefix: str = "run", fmt: str = "xlsx") -> Path:
    """Save a replicate DataFrame under a predictable filename.

    The final path is ``{directory}/{prefix}_{run_id:03d}.{fmt}``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{run_id:03d}.{fmt}"
    if fmt == "xlsx":
        df.to_excel(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")
    return path


def load_replicates(directory: Path, *, pattern: str = "run_*.xlsx") -> list[pd.DataFrame]:
    """Load all replicate files from ``directory`` in sorted order.

    Returns an empty list when no files match (rather than raising), so
    analysis scripts can handle the "no data yet" case cleanly.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            if p.suffix == ".xlsx":
                frames.append(pd.read_excel(p))
            elif p.suffix == ".csv":
                frames.append(pd.read_csv(p))
        except Exception as e:
            log.warning("failed to read %s: %s", p, e)
    return frames


def save_json(obj, path: Path) -> Path:
    """Dump ``obj`` as pretty-printed JSON to ``path``, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_json(path: Path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
