"""Experiment 08 — SynAlign pipeline for perceptual (image) tasks.

Four-stage loop mirroring :mod:`pipeline_symbolic` but for Place Pulse
pairwise judgements.

Stage 1 — generate a Perceptual Mapping Blueprint (from
``synalign.perceptual.stage1_discovery``).
Stage 2 — inject the blueprint into the pairwise prompt and re-run the
pairwise evaluation for each dimension.
Stage 3 — compute Cohen's Kappa against the human Place Pulse choices.
Stage 4 — compare Kappa against the target threshold from ``config.yaml``.
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

from ai4us import client, config, prompts

log = logging.getLogger("synalign.perceptual")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Stage 1 — build the perceptual blueprint
# ---------------------------------------------------------------------------

def stage1_blueprint(llm: client.LLMClient, cfg: dict) -> str:
    log.info("stage 1 — building Perceptual Mapping Blueprint")
    prompt_slug = cfg["perceptual"]["stage1_prompt"]
    body = prompts.get(prompt_slug).body.replace(
        "{N_QUANTITATIVE_ANCHORS}",
        str(cfg["perceptual"]["stage1_samples_per_category"]),
    )
    return llm.chat(body)


# ---------------------------------------------------------------------------
# Stage 2/3 — pairwise judgements with the blueprint injected + kappa
# ---------------------------------------------------------------------------

def stage2_pairwise(llm: client.LLMClient, cfg: dict, blueprint: str,
                    dimensions: list[str], n_pairs: int = 200) -> pd.DataFrame:
    template = cfg["perceptual"]["stage2_prompt_template"]
    # We reuse the Experiment 04 pairwise-evaluation machinery.
    import importlib.util
    helper_path = Path(__file__).resolve().parents[1] / "04_perception_alignment" / "pairwise_eval.py"
    spec = importlib.util.spec_from_file_location("pairwise_eval", helper_path)
    pe = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec.loader is not None
    spec.loader.exec_module(pe)  # type: ignore

    votes = pe._load_votes()  # noqa: SLF001
    results: list[pd.DataFrame] = []
    for dim in dimensions:
        prompt_text = template.format(blueprint=blueprint, dimension=dim)
        # Call the same helper but with our blueprint-augmented prompt.
        # For simplicity we just reuse _parse_choice for robustness.
        subset = votes[votes.get("category") == dim] if "category" in votes.columns else votes
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(subset), size=min(n_pairs, len(subset)), replace=False)
        rows: list[dict] = []
        for k, i in enumerate(idx):
            row = subset.iloc[int(i)]
            left = str(row.get("left_id") or row.get("left") or "")
            right = str(row.get("right_id") or row.get("right") or "")
            try:
                raw = llm.chat_with_image_pair(
                    prompt_text,
                    pe._image_path(left),  # noqa: SLF001
                    pe._image_path(right),  # noqa: SLF001
                )
                ai = pe._parse_choice(raw)  # noqa: SLF001
            except Exception as e:
                log.warning("pair %d: %s", k, e)
                ai = "ERROR"
            rows.append({
                "dimension": dim,
                "left": left,
                "right": right,
                "human_choice": row.get("winner", ""),
                "ai_choice": ai,
            })
        df = pd.DataFrame(rows)
        results.append(df)
        out = (config.paths.experiment_dir("prompt_engineering_perceptual")
               / f"{dim}_with_blueprint.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info("%s: %d judgments -> %s", dim, len(df), out)
    return pd.concat(results, ignore_index=True)


def stage3_kappa(df: pd.DataFrame) -> dict:
    """Per-dimension and overall Cohen's Kappa."""
    result: dict = {}
    for dim, sub in df.groupby("dimension"):
        valid = sub["human_choice"].isin(["LEFT", "RIGHT", "EQUAL"]) & \
                sub["ai_choice"].isin(["LEFT", "RIGHT", "EQUAL"])
        s = sub[valid]
        if len(s) == 0:
            result[dim] = float("nan")
            continue
        po = float((s["human_choice"] == s["ai_choice"]).mean())
        labels = ["LEFT", "RIGHT", "EQUAL"]
        pe = sum(float((s["human_choice"] == lb).mean()
                       * (s["ai_choice"] == lb).mean()) for lb in labels)
        result[dim] = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    result["mean"] = float(np.nanmean(list(result.values())))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-4o-vision")
    parser.add_argument("--dimensions", nargs="*",
                        default=["safety", "beautiful", "lively",
                                 "wealthy", "depressing", "boring"])
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    llm = client.LLMClient(args.model, temperature=0.0)
    blueprint = stage1_blueprint(llm, cfg)
    out_bp = (config.paths.results / "prompt_engineering_perceptual"
              / "perceptual_blueprint.txt")
    out_bp.parent.mkdir(parents=True, exist_ok=True)
    out_bp.write_text(blueprint, encoding="utf-8")

    df = stage2_pairwise(llm, cfg, blueprint, args.dimensions, n_pairs=args.n_pairs)
    kappas = stage3_kappa(df)
    log.info("kappas: %s", kappas)

    target = cfg["perceptual"]["stage3_kappa_target"]
    accept = kappas["mean"] >= target
    summary = pd.DataFrame([
        {"dimension": dim, "kappa": k,
         "accepted": (k is not None and not np.isnan(k) and k >= target)}
        for dim, k in kappas.items() if dim != "mean"
    ])
    out = config.paths.results / "prompt_engineering_perceptual" / "stage4_evaluation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    log.info("accepted overall? %s (threshold %.2f)", accept, target)


if __name__ == "__main__":
    main()
