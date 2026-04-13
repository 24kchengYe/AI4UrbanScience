"""Experiment 04 — have a multimodal model judge Place Pulse pairs.

For every dimension in ``config.yaml``, this script samples ``N`` random
pairs from the Place Pulse 2.0 votes file, asks the AI to choose LEFT /
RIGHT / EQUAL, and writes the per-pair choices to a CSV.

Replaces ``18感知评分-验证AI"感知能力" (pulse随机抽样六维度感知).py``.
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
import random
from pathlib import Path

import pandas as pd
import yaml

from ai4us import client, config, prompts

log = logging.getLogger("perception.pairwise_eval")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _place_pulse_dir() -> Path:
    """Resolve the Place Pulse directory from .env or the default data/ layout."""
    env_dir = config.env("PLACE_PULSE_DIR")
    if env_dir:
        return Path(env_dir)
    return config.paths.data / "place_pulse"


def _load_votes() -> pd.DataFrame:
    pp_dir = _place_pulse_dir()
    votes_csv = pp_dir / "votes.csv"
    if not votes_csv.exists():
        raise FileNotFoundError(
            f"Place Pulse votes file not found at {votes_csv}. "
            f"Set PLACE_PULSE_DIR in .env or download the dataset "
            f"into data/place_pulse/."
        )
    return pd.read_csv(votes_csv)


def _image_path(image_id: str) -> Path:
    return _place_pulse_dir() / "images" / f"{image_id}.jpg"


def _parse_choice(raw: str) -> str:
    token = raw.strip().upper().split()[0] if raw.strip() else "EQUAL"
    token = token.strip(".,!?'\"")
    if token in ("LEFT", "RIGHT", "EQUAL"):
        return token
    # tolerate model variants
    if "LEFT" in token:
        return "LEFT"
    if "RIGHT" in token:
        return "RIGHT"
    return "EQUAL"


def run_one_dimension(llm: client.LLMClient, votes: pd.DataFrame,
                      dimension: str, n_pairs: int, seed: int) -> pd.DataFrame:
    subset = votes[votes["category"] == dimension] if "category" in votes.columns else votes
    rng = random.Random(seed)
    indices = list(range(len(subset)))
    rng.shuffle(indices)

    prompt_template = prompts.get("perception.pairwise_six_dimensions").body
    prompt_text = prompt_template.format(dimension=dimension)

    rows: list[dict] = []
    for i, idx in enumerate(indices[:n_pairs]):
        row = subset.iloc[idx]
        left_id = row.get("left_id") or row.get("left") or row.get("left_image")
        right_id = row.get("right_id") or row.get("right") or row.get("right_image")
        human_choice = row.get("winner") or row.get("choice") or ""
        try:
            raw = llm.chat_with_image_pair(
                prompt_text, _image_path(str(left_id)), _image_path(str(right_id)),
            )
            ai_choice = _parse_choice(raw)
        except Exception as e:
            log.warning("pair %d (%s vs %s): %s", i, left_id, right_id, e)
            ai_choice = "ERROR"

        rows.append({
            "dimension": dimension,
            "left": left_id,
            "right": right_id,
            "human_choice": human_choice,
            "ai_choice": ai_choice,
        })

        if i and i % 50 == 0:
            log.info("[%s] %d/%d", dimension, i, n_pairs)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--n-pairs", type=int, default=cfg["n_pairs_per_dimension"])
    parser.add_argument("--dimensions", nargs="*", default=cfg["dimensions"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    votes = _load_votes()
    llm = client.LLMClient(args.model, temperature=0.0)
    out_dir = config.paths.experiment_dir("perception_alignment") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for dim in args.dimensions:
        df = run_one_dimension(llm, votes, dim, args.n_pairs, cfg["random_seed"])
        out_path = out_dir / f"{dim}.csv"
        df.to_csv(out_path, index=False)
        log.info("%s: %d judgments -> %s", dim, len(df), out_path)


if __name__ == "__main__":
    main()
