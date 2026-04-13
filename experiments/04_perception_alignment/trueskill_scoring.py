"""Experiment 04 — derive TrueSkill-style ratings from pairwise judgments.

Converts the LEFT/RIGHT/EQUAL outcomes in the per-dimension CSVs into image
ratings (a Bayesian skill rating that accommodates uncertainty). Replaces
``18感知评分和trueskill.py``.

Optional dependency::

    pip install trueskill
"""

from __future__ import annotations

# --- path bootstrap so `python experiments/.../foo.py` works without pip install ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# --- end bootstrap ---

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from ai4us import config

try:
    import trueskill  # type: ignore
except ImportError:  # pragma: no cover
    trueskill = None

log = logging.getLogger("perception.trueskill")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rate_dimension(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Run TrueSkill over one dimension's judgments.

    ``source`` picks either ``"human_choice"`` or ``"ai_choice"``.
    """
    if trueskill is None:
        raise RuntimeError("Install trueskill:  pip install trueskill")
    env = trueskill.TrueSkill(draw_probability=0.1)
    ratings: dict[str, trueskill.Rating] = defaultdict(env.create_rating)

    for _, row in df.iterrows():
        left = str(row["left"])
        right = str(row["right"])
        choice = str(row[source]).upper()
        r_left = ratings[left]
        r_right = ratings[right]
        if choice == "LEFT":
            r_left, r_right = env.rate_1vs1(r_left, r_right)
        elif choice == "RIGHT":
            r_right, r_left = env.rate_1vs1(r_right, r_left)
        elif choice == "EQUAL":
            r_left, r_right = env.rate_1vs1(r_left, r_right, drawn=True)
        else:
            continue
        ratings[left] = r_left
        ratings[right] = r_right

    rows: list[dict] = []
    for image_id, rating in ratings.items():
        rows.append({
            "image_id": image_id,
            f"{source}_mu": rating.mu,
            f"{source}_sigma": rating.sigma,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--dimensions", nargs="*", default=cfg["dimensions"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if trueskill is None:
        raise SystemExit("Install trueskill first: pip install trueskill")

    in_dir = config.paths.experiment_dir("perception_alignment") / args.model
    out_dir = config.paths.results / "perception_alignment" / args.model / "trueskill"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in args.dimensions:
        f = in_dir / f"{dim}.csv"
        if not f.exists():
            log.warning("missing %s", f)
            continue
        df = pd.read_csv(f)
        human = rate_dimension(df, "human_choice")
        ai = rate_dimension(df, "ai_choice")
        merged = human.merge(ai, on="image_id", how="outer")
        out = out_dir / f"{dim}.csv"
        merged.to_csv(out, index=False)
        log.info("%s: %d images rated -> %s", dim, len(merged), out)


if __name__ == "__main__":
    main()
