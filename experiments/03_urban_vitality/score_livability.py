"""Experiment 03 step 2 — ask a GenAI model to score each block's livability.

For every replicate produced by :mod:`generate_attributes`, this script sends
the block table back to the model and asks it to return a JSON object mapping
block names to livability scores in ``[0, 1]``. The resulting scores are
joined back to the attribute table and saved to
``data/generated/vitality_scored/<model>/<scoring_prompt>/``.
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
import time
from pathlib import Path

import pandas as pd
import yaml

from ai4us import client, config, io, prompts

log = logging.getLogger("vitality.score_livability")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _format_block_table(df: pd.DataFrame) -> str:
    """Serialise a block attribute table as the text block the scorer expects."""
    lines = [", ".join(df.columns.astype(str))]
    for _, row in df.iterrows():
        lines.append(", ".join(str(v) for v in row.values))
    return "\n".join(lines)


def score_one_replicate(llm: client.LLMClient, scoring_prompt: str,
                        attribute_df: pd.DataFrame) -> pd.DataFrame:
    """Send one replicate to the scorer and return a merged frame."""
    table_str = _format_block_table(attribute_df)
    message = f"{scoring_prompt}\n\nBlock data:\n{table_str}"
    raw = llm.chat(message)
    scores = io.parse_json_map(raw)

    # Merge back into the attribute frame.
    scored = attribute_df.copy()
    scored["Livability"] = scored[attribute_df.columns[0]].map(scores).astype(float)
    return scored


def run_all(model_key: str, attribute_prompt: str, scoring_prompt: str,
            *, temperature: float, delay_s: float, resume: bool = True) -> Path:
    llm = client.LLMClient(model_key, temperature=temperature)
    prompt = prompts.get(scoring_prompt)

    in_dir = (config.paths.experiment_dir("vitality_attributes")
              / model_key / attribute_prompt)
    out_dir = (config.paths.experiment_dir("vitality_scored")
               / model_key / scoring_prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(in_dir.glob("run_*.xlsx"))
    if not input_files:
        raise FileNotFoundError(
            f"No attribute replicates under {in_dir}. "
            f"Run generate_attributes.py first."
        )
    log.info("Scoring %d replicates from %s / %s -> %s",
             len(input_files), model_key, attribute_prompt, out_dir)

    for i, f in enumerate(input_files, start=1):
        run_id = int(f.stem.split("_")[-1])
        target = out_dir / f"run_{run_id:03d}.xlsx"
        if resume and target.exists():
            log.info("[%d/%d] skip existing", i, len(input_files))
            continue
        try:
            attrs = pd.read_excel(f)
            scored = score_one_replicate(llm, prompt.body, attrs)
            scored.to_excel(target, index=False)
            log.info("[%d/%d] %s: %d scored",
                     i, len(input_files), run_id, scored["Livability"].notna().sum())
        except Exception as e:  # noqa: BLE001
            log.exception("[%d/%d] failed: %s", i, len(input_files), e)
        if i < len(input_files) and delay_s > 0:
            time.sleep(delay_s)

    return out_dir


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--attribute-prompt", default=cfg["primary_attribute_prompt"])
    parser.add_argument("--scoring-prompt", default=cfg["primary_scoring_prompt"])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_all(
        args.model, args.attribute_prompt, args.scoring_prompt,
        temperature=cfg["temperature"],
        delay_s=cfg["request_delay_s"],
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
