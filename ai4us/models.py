"""Registry of all LLM / MLLM models used in the paper.

Each entry maps a human-readable model key to the concrete HTTP endpoint and
authentication source that should be used when calling it. Experiment scripts
refer to models by key (e.g. ``"gpt-4o"``) and never hard-code endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ai4us import config


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a single LLM/MLLM endpoint."""

    key: str
    """Short identifier used in configs and filenames."""

    display_name: str
    """Human-readable name printed in plots and logs."""

    provider: str
    """Which provider/router is used to reach the model."""

    remote_model: str
    """The ``model`` field sent in the API request body."""

    endpoint_fn: Callable[[], str]
    """Function returning the base URL at call time (so env is read lazily)."""

    key_fn: Callable[[], str]
    """Function returning the API key at call time."""

    supports_system_role: bool = True
    """Some models (e.g. ``o1-preview``) reject a ``system`` message."""

    is_multimodal: bool = False
    """Whether the endpoint can accept image inputs."""


# ---------------------------------------------------------------------------
# Text models used for symbolic experiments (scaling, decay, vitality)
# ---------------------------------------------------------------------------

TEXT_MODELS: dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec(
        key="gpt-4o",
        display_name="GPT-4o",
        provider="gptsapi",
        remote_model="gpt-4o",
        endpoint_fn=config.gptsapi_url,
        key_fn=config.gptsapi_key,
    ),
    "gpt-4o-mindcraft": ModelSpec(
        key="gpt-4o-mindcraft",
        display_name="GPT-4o (MindCraft)",
        provider="mindcraft",
        remote_model="gpt-4o",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
    ),
    "o1-preview": ModelSpec(
        key="o1-preview",
        display_name="o1-preview",
        provider="mindcraft",
        remote_model="o1-preview",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
        supports_system_role=False,
    ),
    "claude-3.5-sonnet": ModelSpec(
        key="claude-3.5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider="gptsapi",
        remote_model="claude-3-5-sonnet-20241022",
        endpoint_fn=config.gptsapi_url,
        key_fn=config.gptsapi_key,
    ),
    "deepseek-v3": ModelSpec(
        key="deepseek-v3",
        display_name="DeepSeek V3",
        provider="deepseek",
        remote_model="deepseek-chat",
        endpoint_fn=config.deepseek_url,
        key_fn=config.deepseek_key,
    ),
    "chatglm-4": ModelSpec(
        key="chatglm-4",
        display_name="ChatGLM 4",
        provider="bigmodel",
        remote_model="glm-4-0520",
        endpoint_fn=config.chatglm_url,
        key_fn=config.chatglm_key,
    ),
    "gemini-2.0-flash": ModelSpec(
        key="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        provider="mindcraft",
        remote_model="gemini-2.0-flash-exp",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
    ),
    "qwen-plus": ModelSpec(
        key="qwen-plus",
        display_name="Qwen Plus",
        provider="mindcraft",
        remote_model="qwen-plus-latest",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
    ),
    "doubao-pro": ModelSpec(
        key="doubao-pro",
        display_name="Doubao Pro 128k",
        provider="mindcraft",
        remote_model="Doubao-pro-128k",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
    ),
    "hunyuan-large": ModelSpec(
        key="hunyuan-large",
        display_name="Hunyuan Large",
        provider="mindcraft",
        remote_model="hunyuan-large-longcontext",
        endpoint_fn=config.mindcraft_url,
        key_fn=config.mindcraft_key,
    ),
}


# ---------------------------------------------------------------------------
# Multimodal models used for perception experiments (Fig.3 / Fig.6)
# ---------------------------------------------------------------------------

MULTIMODAL_MODELS: dict[str, ModelSpec] = {
    "gpt-4o-vision": ModelSpec(
        key="gpt-4o-vision",
        display_name="GPT-4o (vision)",
        provider="ablai",
        remote_model="gpt-4o",
        endpoint_fn=lambda: f"{config.ablai_base_url()}/chat/completions",
        key_fn=config.ablai_key,
        is_multimodal=True,
    ),
    "nano-banana": ModelSpec(
        key="nano-banana",
        display_name="Nano Banana (image gen)",
        provider="ablai",
        remote_model="nano-banana",
        endpoint_fn=lambda: f"{config.ablai_base_url()}/images/generations",
        key_fn=config.ablai_key,
        is_multimodal=True,
    ),
    "nano-banana-edit": ModelSpec(
        key="nano-banana-edit",
        display_name="Nano Banana (image edit)",
        provider="ablai",
        remote_model="nano-banana",
        endpoint_fn=lambda: f"{config.ablai_base_url()}/images/edits",
        key_fn=config.ablai_key,
        is_multimodal=True,
    ),
}


ALL_MODELS: dict[str, ModelSpec] = {**TEXT_MODELS, **MULTIMODAL_MODELS}


def get(key: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by its short key."""
    try:
        return ALL_MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model key {key!r}. Available models: "
            f"{sorted(ALL_MODELS)}"
        ) from exc


def text_model_keys() -> list[str]:
    """Return the list of keys for all text models (stable, insertion order)."""
    return list(TEXT_MODELS.keys())
