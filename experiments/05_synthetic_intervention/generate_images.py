"""Experiment 05 — generate baseline images and intervention variants.

Replaces the two legacy scripts:
  * ``17视觉感知实验-0纯图片生成（baseline）.py``
  * ``17视觉感知实验-0纯图片生成（图改图 干预变体）.py``

Example::

    python experiments/05_synthetic_intervention/generate_images.py --mode baseline
    python experiments/05_synthetic_intervention/generate_images.py --mode variants
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

import requests
import yaml

from ai4us import client, config, prompts

log = logging.getLogger("intervention.generate_images")
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _out_dir(sub: str) -> Path:
    return config.paths.experiment_dir("perception_images") / sub


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


# ---------------------------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------------------------

def _random_variables() -> dict:
    rng = random.Random()
    return {
        "region": rng.choice(["North America", "Europe", "East Asia", "Latin America",
                              "Middle East", "Africa", "Southeast Asia"]),
        "time_of_day": rng.choice(["early morning", "midday", "late afternoon",
                                   "golden hour", "evening"]),
        "density": rng.choice(["sparse", "moderate", "dense", "ultra-dense"]),
        "land_use": rng.choice(["residential", "commercial", "industrial", "mixed-use"]),
        "vegetation": rng.choice(["none", "sparse", "moderate", "lush"]),
        "road_type": rng.choice(["pedestrian street", "boulevard", "alley",
                                 "highway overpass", "bike-friendly street"]),
    }


def generate_baseline(n: int) -> list[Path]:
    prompt = prompts.get("image.structured_diverse")
    gen = client.ImageGenClient()
    out_dir = _out_dir("baseline")
    paths: list[Path] = []
    for i in range(1, n + 1):
        text = prompt.body.format(**_random_variables())
        try:
            urls = gen.generate(text, n=1)
            if not urls:
                continue
            target = out_dir / f"baseline_{i:03d}.jpg"
            _download(urls[0], target)
            paths.append(target)
            log.info("[%d/%d] saved %s", i, n, target.name)
        except Exception as e:  # noqa: BLE001
            log.exception("[%d/%d] failed: %s", i, n, e)
    return paths


# ---------------------------------------------------------------------------
# Intervention variants
# ---------------------------------------------------------------------------

def generate_variants(n_pairs_per_category: int) -> None:
    cfg = _load_config()
    gen = client.ImageGenClient()

    baseline_dir = _out_dir("baseline")
    baselines = sorted(baseline_dir.glob("baseline_*.jpg"))
    if not baselines:
        raise FileNotFoundError(
            f"No baseline images at {baseline_dir}; run with --mode baseline first."
        )

    for category, prompt_slug in cfg["intervention_categories"].items():
        prompt = prompts.get(prompt_slug)
        out_dir = _out_dir(f"variants/{category}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, base in enumerate(baselines[:n_pairs_per_category], start=1):
            try:
                urls = gen.edit(prompt.body, base, n=1)
                if not urls:
                    continue
                target = out_dir / f"{base.stem}__{category}.jpg"
                _download(urls[0], target)
                log.info("[%s %d/%d] %s", category, i, n_pairs_per_category, target.name)
            except Exception as e:  # noqa: BLE001
                log.exception("[%s %d] failed: %s", category, i, e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["baseline", "variants"], required=True)
    parser.add_argument("--n-baseline", type=int, default=cfg["n_baseline_images"])
    parser.add_argument("--n-pairs", type=int, default=cfg["n_pairs_per_category"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.mode == "baseline":
        generate_baseline(args.n_baseline)
    else:
        generate_variants(args.n_pairs)


if __name__ == "__main__":
    main()
