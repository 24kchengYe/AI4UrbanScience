"""Experiment 08 — SynAlign pipeline for symbolic generation tasks.

Four-stage loop:
  1. Discovery   — ask the model to synthesise a statistical blueprint
  2. Synthesis   — generate replicate datasets with the blueprint injected
  3. Alignment   — fit the theoretical law and compare against the blueprint
  4. Evaluation  — decide whether to accept the blueprint or loop again

Replaces the four ``16参考SynAlign01…04`` scripts plus the
``SynAlign00-synthesis-人类知识注入提示增强版本 symbolic*.py`` set (single
version + three parallel variants).
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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai4us import client, config, io, prompts

log = logging.getLogger("synalign.symbolic")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Import scaling-law analyzer lazily to avoid a hard package dependency.
def _fit_one_replicate(df: pd.DataFrame) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_urban_scaling_law"))
    import analyze as scaling_analyze  # type: ignore
    return scaling_analyze.fit_one_replicate(df)


# ---------------------------------------------------------------------------
# Stage 1 — discovery
# ---------------------------------------------------------------------------

def stage1_discovery(llm: client.LLMClient, cfg: dict) -> str:
    log.info("stage 1 — synthesising statistical blueprint")
    return llm.chat(cfg["symbolic"]["stage1_discovery_prompt"])


# ---------------------------------------------------------------------------
# Stage 2 — synthesis with blueprint
# ---------------------------------------------------------------------------

def stage2_synthesis(llm: client.LLMClient, cfg: dict,
                     blueprint: str, *, use_blueprint: bool) -> Path:
    base_prompt_slug = cfg["symbolic"]["stage2_base_prompt"]
    bp_prompt_slug = cfg["symbolic"]["stage2_blueprint_prompt"]
    base = prompts.get(base_prompt_slug).body
    template = prompts.get(bp_prompt_slug).body
    prompt_text = template + "\n\nBlueprint:\n" + blueprint if use_blueprint else base

    mode = "with_blueprint" if use_blueprint else "without_blueprint"
    out_dir = (config.paths.experiment_dir("prompt_engineering_symbolic")
               / mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["City Name", "Population", "Infrastructure", "GDP"]
    n = cfg["n_samples_per_stage"]
    log.info("stage 2 — generating %d replicates (%s)", n, mode)

    for i in range(1, n + 1):
        target = out_dir / f"run_{i:03d}.xlsx"
        if target.exists():
            continue
        try:
            raw = llm.chat(prompt_text)
            df = io.parse_delimited(raw, columns)
            io.save_replicate(df, out_dir, i)
        except Exception as e:  # noqa: BLE001
            log.exception("[%d/%d] %s", i, n, e)
    return out_dir


# ---------------------------------------------------------------------------
# Stage 3 — alignment: fit each replicate and compute the chosen metric
# ---------------------------------------------------------------------------

def stage3_alignment(replicate_dir: Path) -> pd.DataFrame:
    log.info("stage 3 — fitting replicates from %s", replicate_dir)
    rows: list[dict] = []
    for f in sorted(replicate_dir.glob("run_*.xlsx")):
        try:
            fit = _fit_one_replicate(pd.read_excel(f))
            fit["file"] = f.name
            rows.append(fit)
        except Exception as e:
            log.warning("skip %s: %s", f.name, e)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4 — evaluation: accept blueprint if R² ≥ threshold in majority
# ---------------------------------------------------------------------------

def stage4_evaluation(fits: pd.DataFrame, cfg: dict) -> dict:
    metric = cfg["symbolic"]["stage3_target"]
    threshold = cfg["symbolic"]["stage4_acceptance_threshold"]
    if metric not in fits.columns:
        return {"accept": False, "reason": f"metric {metric} missing from fits"}

    values = fits[metric].dropna()
    if not len(values):
        return {"accept": False, "reason": "no successful fits"}

    pass_rate = float((values >= 0.85).mean())
    accept = pass_rate >= threshold
    return {
        "accept": bool(accept),
        "pass_rate": pass_rate,
        "mean_metric": float(values.mean()),
        "n": int(len(values)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the without-blueprint baseline arm.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    llm = client.LLMClient(args.model, temperature=0.0)

    blueprint = stage1_discovery(llm, cfg)
    io.save_json({"blueprint_text": blueprint},
                 config.paths.results / "prompt_engineering_symbolic" / "blueprint.json")

    # With blueprint
    bp_dir = stage2_synthesis(llm, cfg, blueprint, use_blueprint=True)
    bp_fits = stage3_alignment(bp_dir)
    bp_eval = stage4_evaluation(bp_fits, cfg)
    log.info("with blueprint: %s", bp_eval)

    # Without blueprint (baseline ablation)
    if not args.skip_baseline:
        base_dir = stage2_synthesis(llm, cfg, blueprint, use_blueprint=False)
        base_fits = stage3_alignment(base_dir)
        base_eval = stage4_evaluation(base_fits, cfg)
        log.info("without blueprint: %s", base_eval)

        # Save the ablation summary
        summary = pd.DataFrame([
            {"arm": "with_blueprint", **bp_eval},
            {"arm": "without_blueprint", **base_eval},
        ])
        out = config.paths.results / "prompt_engineering_symbolic" / "ablation.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        log.info("ablation -> %s", out)


if __name__ == "__main__":
    main()
