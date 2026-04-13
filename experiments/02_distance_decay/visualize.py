"""Experiment 02 — visualize generated distance-decay profiles.

This produces the supplementary distance-decay figure: a panel showing the
100 replicate density curves (grey) plus the mean inverse-S fit (red) and
the theoretical reference (black).
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ai4us import config, fitting, theories, viz

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_curves(model: str, prompt: str) -> list[tuple[np.ndarray, np.ndarray]]:
    in_dir = config.paths.model_dir("distance_decay", model, prompt)
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for f in sorted(in_dir.glob("run_*.xlsx")):
        try:
            df = pd.read_excel(f)
            r = pd.to_numeric(
                df["Circle Layer"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            ).values.astype(float)
            d = pd.to_numeric(
                df["Land Density"].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
                errors="coerce",
            ).values.astype(float)
            mask = np.isfinite(r) & np.isfinite(d)
            if mask.sum() < 5:
                continue
            curves.append((r[mask], d[mask] / (np.nanmax(d[mask]) or 1.0)))
        except Exception:
            continue
    return curves


def main(argv: list[str] | None = None) -> None:
    cfg = _load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg["primary_model"])
    parser.add_argument("--prompt", default=cfg["primary_prompt"])
    parser.add_argument("--output", type=Path,
                        default=config.paths.figures / "distance_decay.png")
    args = parser.parse_args(argv)

    viz.use_paper_style(font_size=12)
    curves = _load_curves(args.model, args.prompt)
    if not curves:
        raise SystemExit(
            f"no replicate curves found; run `generate.py --model {args.model} "
            f"--prompt {args.prompt}` first"
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    for r, d in curves:
        ax.plot(r, d, color=viz.PALETTE["gray"], alpha=0.1, linewidth=0.7)

    all_r = np.concatenate([c[0] for c in curves])
    all_d = np.concatenate([c[1] for c in curves])
    master = fitting.fit_inverse_s(all_r, all_d)
    r_grid = np.linspace(all_r.min(), all_r.max(), 200)
    ax.plot(r_grid, master.predict(r_grid),
            color=viz.PALETTE["accent"], linewidth=2.5,
            label=(f"mean fit: r0={master.r0:.1f}, "
                   f"α={master.alpha:.2f}, β={master.beta:.2f}, "
                   f"R²={master.r_squared:.3f}"))

    ax.set_xlabel("distance from center (ring index)")
    ax.set_ylabel("normalized land density")
    ax.set_title(f"Distance decay on {args.model} ({len(curves)} replicates)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    viz.save_figure(fig, args.output, formats=("png", "pdf"))
    print(f"figure saved to {args.output}")


if __name__ == "__main__":
    main()
