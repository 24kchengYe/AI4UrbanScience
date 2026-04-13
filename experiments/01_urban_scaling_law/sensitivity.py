"""Experiment 01 — sensitivity analysis: independent vs joint sampling.

This module replaces the seven legacy ``01test-sensitivity-*`` scripts
(independent sampling, joint sampling, and various visualisations of each)
with a single entry point.

Two sampling paradigms are compared:

* **independent** — one city per API request. Each request is stateless
  and the model cannot reuse context across cities. This emulates the
  uncontrolled generation regime discussed in the manuscript's *Prompt
  Engineering* section.
* **joint** — one request returns all ``batch_size`` cities in one shot.
  The model sees every previous city when it decides on the next one,
  which activates its spatial-logic priors.

The script writes replicate files to
``data/generated/scaling_law/<model>/sensitivity/<mode>_bs{batch_size}/``.
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

from ai4us import client, config, io, models, prompts

log = logging.getLogger("scaling_law.sensitivity")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Single-city prompt for the independent mode
# ---------------------------------------------------------------------------

_ONE_CITY_PROMPT = (
    "Generate ONE city record in a country.\n"
    "Format the output as:\n"
    "CityName, Population, Infrastructure volume, GDP\n"
    "Output exactly one line without any additional text or explanations.\n"
    "- Infrastructure volume: total road miles\n"
    "- GDP: all Gross Domestic Product of the city in one year\n"
)


# ---------------------------------------------------------------------------
# Sampling modes
# ---------------------------------------------------------------------------

def _save_to_mode_dir(df: pd.DataFrame, model: str, mode: str,
                     batch_size: int, run_id: int) -> Path:
    out_dir = (config.paths.experiment_dir("scaling_law")
               / model / "sensitivity" / f"{mode}_bs{batch_size}")
    return io.save_replicate(df, out_dir, run_id)


def run_independent(llm: client.LLMClient, columns: list[str],
                    *, n_cities: int, delay_s: float) -> pd.DataFrame:
    """Generate ``n_cities`` one-at-a-time and concatenate the results."""
    rows: list[pd.DataFrame] = []
    for i in range(n_cities):
        raw = llm.chat(_ONE_CITY_PROMPT)
        df = io.parse_delimited(raw, columns)
        if len(df):
            rows.append(df.iloc[[0]])  # keep just the first valid line
        if delay_s > 0:
            time.sleep(delay_s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def run_joint(llm: client.LLMClient, prompt_text: str, columns: list[str]) -> pd.DataFrame:
    raw = llm.chat(prompt_text)
    return io.parse_delimited(raw, columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"],
                        help="Prompt slug (used only in joint mode).")
    parser.add_argument("--mode", choices=["independent", "joint"], required=True)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Cities per replicate. For independent mode this is "
                             "the number of individual API calls per replicate.")
    parser.add_argument("--n-replicates", type=int, default=cfg["n_replicates"])
    parser.add_argument("--delay-s", type=float, default=cfg["request_delay_s"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    llm = client.LLMClient(args.model, temperature=cfg["temperature"])
    columns = cfg["columns"]
    prompt_text = prompts.get(args.prompt).body if args.mode == "joint" else ""

    for run_id in range(1, args.n_replicates + 1):
        if args.mode == "independent":
            df = run_independent(llm, columns,
                                 n_cities=args.batch_size,
                                 delay_s=args.delay_s)
        else:
            df = run_joint(llm, prompt_text, columns)

        path = _save_to_mode_dir(df, args.model, args.mode,
                                 args.batch_size, run_id)
        log.info("[%d/%d] mode=%s bs=%d rows=%d -> %s",
                 run_id, args.n_replicates, args.mode, args.batch_size,
                 len(df), path)

        if run_id < args.n_replicates and args.delay_s > 0:
            time.sleep(args.delay_s)


if __name__ == "__main__":
    main()
