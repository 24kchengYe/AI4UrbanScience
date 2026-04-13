"""Experiment 05 — validate diversity of generated images via CLIP embeddings.

Replaces ``17视觉感知实验-1CLIP验证多样性.py``.

The metric reported in the paper is the mean cosine distance between every
pair of generated images (the higher the better).
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
from PIL import Image

try:
    import torch
    import clip  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    clip = None

from ai4us import config

log = logging.getLogger("intervention.clip_diversity")


def _load_clip():
    if clip is None or torch is None:
        raise RuntimeError(
            "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def embed_directory(img_dir: Path) -> np.ndarray:
    model, preprocess, device = _load_clip()
    vectors: list[np.ndarray] = []
    for f in sorted(img_dir.glob("*.jpg")):
        try:
            img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                v = model.encode_image(img).cpu().numpy().flatten()
            v = v / (np.linalg.norm(v) + 1e-9)
            vectors.append(v)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", f.name, e)
    return np.stack(vectors) if vectors else np.empty((0, 0))


def mean_pairwise_distance(vectors: np.ndarray) -> float:
    n = len(vectors)
    if n < 2:
        return float("nan")
    sims = vectors @ vectors.T
    mask = np.triu(np.ones_like(sims), k=1).astype(bool)
    cos_sims = sims[mask]
    return float(np.mean(1.0 - cos_sims))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-dir", type=Path,
                        default=config.paths.experiment_dir("perception_images") / "baseline",
                        help="Directory containing the images to evaluate.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    vectors = embed_directory(args.image_dir)
    d = mean_pairwise_distance(vectors)
    log.info("embeddings: %d images in %s", len(vectors), args.image_dir)
    print(f"mean pairwise CLIP cosine distance: {d:.4f}")


if __name__ == "__main__":
    main()
