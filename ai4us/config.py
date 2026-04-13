"""Project-wide configuration: paths and environment variables.

All directory layout decisions live here so experiment scripts stay path-free.
Paths are relative to the repository root and created on demand.

Environment variables are loaded from a `.env` file at the repo root the first
time this module is imported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; users can export variables manually instead.
    pass


# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from this file until we find the repository marker."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback: assume the parent of the `ai4us` package is the root.
    return here.parent.parent


REPO_ROOT: Path = _find_repo_root()


# ---------------------------------------------------------------------------
# Standard directory layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Canonical paths for the project. Every value is a :class:`pathlib.Path`."""

    repo_root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    generated: Path = REPO_ROOT / "data" / "generated"
    real_world: Path = REPO_ROOT / "data" / "real_world"
    place_pulse: Path = REPO_ROOT / "data" / "place_pulse"
    results: Path = REPO_ROOT / "results"
    figures: Path = REPO_ROOT / "results" / "figures"
    logs: Path = REPO_ROOT / "results" / "logs"

    def experiment_dir(self, experiment: str) -> Path:
        """Return the per-experiment generated-data directory."""
        return self.generated / experiment

    def model_dir(self, experiment: str, model: str, prompt_id: str | None = None) -> Path:
        """Return the per-experiment / per-model / per-prompt output directory."""
        d = self.experiment_dir(experiment) / model
        if prompt_id:
            d = d / prompt_id
        return d

    def ensure(self, *paths: Path) -> None:
        """Create the given directories (and all parents) if they do not exist."""
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)


paths = Paths()


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    """Return an environment variable value.

    Parameters
    ----------
    name : str
        Variable name.
    default : str or None
        Fallback value if the variable is missing or empty.
    required : bool
        When ``True``, raise :class:`RuntimeError` if neither the variable nor
        a default is available.
    """
    value = os.getenv(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if required:
        raise RuntimeError(
            f"Environment variable {name!r} is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return ""


# Convenience accessors for the most-used secrets and endpoints.

def mindcraft_key() -> str:
    return env("MINDCRAFT_KEY", required=True)


def mindcraft_url() -> str:
    return env("MINDCRAFT_BASE_URL",
               default="https://api.mindcraft.com.cn/v1/chat/completions")


def gptsapi_key() -> str:
    return env("GPTSAPI_KEY", required=True)


def gptsapi_url() -> str:
    return env("GPTSAPI_BASE_URL",
               default="https://api.gptsapi.net/v1/chat/completions")


def deepseek_key() -> str:
    return env("DEEPSEEK_KEY", required=True)


def deepseek_url() -> str:
    return env("DEEPSEEK_BASE_URL",
               default="https://api.deepseek.com/chat/completions")


def chatglm_key() -> str:
    return env("CHATGLM_KEY", required=True)


def chatglm_url() -> str:
    return env("BIGMODEL_BASE_URL",
               default="https://open.bigmodel.cn/api/paas/v4/chat/completions")


def ablai_key() -> str:
    return env("ABLAI_KEY", required=True)


def ablai_base_url() -> str:
    return env("ABLAI_BASE_URL", default="https://api.ablai.top/v1")
