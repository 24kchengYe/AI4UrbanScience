"""AI4UrbanScience — core library for the AI4US framework.

Modules
-------
config      Environment variables and standardized directory layout.
models      Metadata registry for every LLM/MLLM used in the paper.
client      Unified HTTP client for text/image/multimodal APIs.
prompts     Registry of every prompt variant used across experiments.
io          Helpers for reading/writing generated datasets (xlsx/csv/json).
theories    Mathematical forms of the urban theories we test.
fitting     Power-law, inverse-S and OLS fitters used across experiments.
metrics     Distribution-divergence metrics: MAE, Overlap Ratio, JSD.
viz         Shared plotting style and small utilities.
"""

from ai4us import (
    config,
    models,
    client,
    prompts,
    io,
    theories,
    fitting,
    metrics,
    viz,
)

__version__ = "1.0.0"

__all__ = [
    "config",
    "models",
    "client",
    "prompts",
    "io",
    "theories",
    "fitting",
    "metrics",
    "viz",
]
