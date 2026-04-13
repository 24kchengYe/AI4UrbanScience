"""Shared plotting style and small plot helpers.

A single place to configure fonts, color palettes and axes so that every
figure in the paper looks consistent. Experiment scripts call
``viz.use_paper_style()`` once at the top of the figure script.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# The paper uses a fixed palette so every figure is visually consistent.
PALETTE = {
    "real":      "#2166ac",   # blue
    "generated": "#ef8a62",   # orange
    "theory":    "#1a1a1a",   # near-black
    "empirical": "#4393c3",   # lighter blue for benchmarks
    "accent":    "#d6604d",   # warm red for fit lines
    "gray":      "#888888",
}


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def use_paper_style(*, font_size: int = 12) -> None:
    """Apply the paper-wide Matplotlib style.

    Call this once at the top of a figure script. Subsequent
    ``plt.subplots`` calls will inherit the style.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


@contextmanager
def paper_style(**kwargs):
    """Temporarily apply the paper style to a single figure."""
    original = plt.rcParams.copy()
    try:
        use_paper_style(**kwargs)
        yield
    finally:
        plt.rcParams.update(original)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(fig, path: Path, *, dpi: int = 300, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Save ``fig`` to one or more files and return the resulting paths.

    Useful for emitting both a PNG preview and a PDF for publication.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        outputs.append(out)
    return outputs
