"""Experiment 01 — generate urban scaling-law datasets with a GenAI model.

Replaces the original ten model-specific scripts
(``01test-openai(Zipf-Scaling).py`` and friends) with a single entry point
whose behaviour is fully controlled by CLI flags and ``config.yaml``.

Examples
--------
Generate 100 replicates from the primary model with the baseline prompt::

    python experiments/01_urban_scaling_law/generate.py

Run a specific model/prompt combination::

    python experiments/01_urban_scaling_law/generate.py \\
        --model claude-3.5-sonnet --prompt scaling_law.with_theory

Run every model in the config in sequence::

    python experiments/01_urban_scaling_law/generate.py --all-models
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

import yaml

from ai4us import client, config, io, models, prompts

log = logging.getLogger("scaling_law.generate")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(
    llm: client.LLMClient,
    prompt_text: str,
    columns: list[str],
    *,
    min_rows: int | None = None,
) -> "pd.DataFrame":
    """Call the LLM once and parse the response into a DataFrame."""
    import pandas as pd  # local import so the CLI `--help` works without pandas
    raw = llm.chat(prompt_text)
    df = io.parse_delimited(raw, columns)
    if min_rows is not None and len(df) < min_rows:
        log.warning("parsed only %d rows (expected >= %d)", len(df), min_rows)
    return df


def generate_replicates(
    model_key: str,
    prompt_slug: str,
    *,
    n_replicates: int,
    columns: list[str],
    n_cities_per_replicate: int,
    delay_s: float,
    temperature: float,
    resume: bool = True,
) -> Path:
    """Generate ``n_replicates`` datasets for a single (model, prompt) pair.

    Returns the output directory (under ``data/generated/scaling_law/...``).
    The function is idempotent: already-existing replicate files are skipped
    when ``resume=True`` so you can safely resume after a rate-limit failure.
    """
    spec = models.get(model_key)
    prompt = prompts.get(prompt_slug)
    out_dir = config.paths.model_dir("scaling_law", model_key, prompt_slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = client.LLMClient(model_key, temperature=temperature)

    log.info(
        "Generating %d replicates of prompt %r with model %r → %s",
        n_replicates, prompt_slug, spec.display_name, out_dir,
    )

    for i in range(1, n_replicates + 1):
        target = out_dir / f"run_{i:03d}.xlsx"
        if resume and target.exists():
            log.info("[%d/%d] skipping existing file", i, n_replicates)
            continue
        try:
            df = generate_one(llm, prompt.body, columns,
                              min_rows=int(0.5 * n_cities_per_replicate))
            io.save_replicate(df, out_dir, i)
            log.info("[%d/%d] %d rows saved", i, n_replicates, len(df))
        except client.APIError as e:
            log.error("[%d/%d] API error: %s", i, n_replicates, e)
        except Exception as e:  # noqa: BLE001 — we want the loop to continue
            log.exception("[%d/%d] unexpected error: %s", i, n_replicates, e)

        if i < n_replicates and delay_s > 0:
            time.sleep(delay_s)

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate urban scaling-law datasets with a GenAI model.",
    )
    p.add_argument("--model", default=None,
                   help="Model key (default: value of primary_model in config.yaml)")
    p.add_argument("--all-models", action="store_true",
                   help="Iterate over every model listed in config.all_models.")
    p.add_argument("--prompt", default=None,
                   help="Prompt slug (default: value of primary_prompt in config.yaml)")
    p.add_argument("--all-prompts", action="store_true",
                   help="Iterate over every prompt in config.prompt_variants.")
    p.add_argument("--n-replicates", type=int, default=None,
                   help="Override n_replicates from config.yaml.")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-generate replicates even if the output file already exists.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config()

    model_keys = (cfg["all_models"] if args.all_models
                  else [args.model or cfg["primary_model"]])
    prompt_slugs = (cfg["prompt_variants"] if args.all_prompts
                    else [args.prompt or cfg["primary_prompt"]])
    n_replicates = args.n_replicates or cfg["n_replicates"]
    columns = cfg["columns"]

    log.info("plan: %d model(s) × %d prompt(s) × %d replicate(s)",
             len(model_keys), len(prompt_slugs), n_replicates)

    for model_key in model_keys:
        for prompt_slug in prompt_slugs:
            generate_replicates(
                model_key, prompt_slug,
                n_replicates=n_replicates,
                columns=columns,
                n_cities_per_replicate=cfg["n_cities_per_replicate"],
                delay_s=cfg["request_delay_s"],
                temperature=cfg["temperature"],
                resume=not args.no_resume,
            )


if __name__ == "__main__":
    main()
