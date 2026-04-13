"""Experiment 02 — generate distance-decay profiles with a GenAI model.

Replaces the five legacy ``07test-<model>(distance).py`` scripts.

Examples
--------
    python experiments/02_distance_decay/generate.py
    python experiments/02_distance_decay/generate.py --model claude-3.5-sonnet
    python experiments/02_distance_decay/generate.py --all-models --all-prompts
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

log = logging.getLogger("distance_decay.generate")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def generate_replicates(model_key: str, prompt_slug: str,
                        *, n_replicates: int, columns: list[str],
                        delay_s: float, temperature: float,
                        resume: bool = True) -> Path:
    prompt = prompts.get(prompt_slug)
    llm = client.LLMClient(model_key, temperature=temperature)
    out_dir = config.paths.model_dir("distance_decay", model_key, prompt_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating %d replicates for %s / %s", n_replicates, model_key, prompt_slug)

    for i in range(1, n_replicates + 1):
        target = out_dir / f"run_{i:03d}.xlsx"
        if resume and target.exists():
            log.info("[%d/%d] skip existing", i, n_replicates)
            continue
        try:
            raw = llm.chat(prompt.body)
            df = io.parse_delimited(raw, columns)
            io.save_replicate(df, out_dir, i)
            log.info("[%d/%d] rows=%d", i, n_replicates, len(df))
        except Exception as e:  # noqa: BLE001
            log.exception("[%d/%d] failed: %s", i, n_replicates, e)
        if i < n_replicates and delay_s > 0:
            time.sleep(delay_s)

    return out_dir


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_keys = cfg["all_models"] if args.all_models else [args.model or cfg["primary_model"]]
    prompt_slugs = cfg["prompt_variants"] if args.all_prompts else [args.prompt or cfg["primary_prompt"]]
    n_replicates = args.n_replicates or cfg["n_replicates"]

    for m in model_keys:
        for p in prompt_slugs:
            generate_replicates(
                m, p,
                n_replicates=n_replicates,
                columns=cfg["columns"],
                delay_s=cfg["request_delay_s"],
                temperature=cfg["temperature"],
                resume=not args.no_resume,
            )


if __name__ == "__main__":
    main()
