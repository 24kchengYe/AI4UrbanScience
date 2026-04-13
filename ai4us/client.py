"""Unified LLM/MLLM HTTP client.

All experiment scripts call LLMs through :class:`LLMClient`. The client looks
up endpoint URL, credentials and the ``model`` string from
:mod:`ai4us.models`, so experiment code never touches raw HTTP.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ai4us import models

log = logging.getLogger(__name__)


class APIError(RuntimeError):
    """Raised when an API call returns a non-2xx status code."""


@dataclass
class LLMClient:
    """Thin wrapper around ``requests.post`` that handles the three things
    every experiment would otherwise re-implement:

    * choosing the right endpoint + API key for a model,
    * retrying on transient failures,
    * parsing the ``choices[0].message.content`` envelope.

    Examples
    --------
    >>> client = LLMClient("gpt-4o")
    >>> text = client.chat("Return 3 random US city names as a comma-separated list.")
    """

    model: str
    temperature: float = 0.0
    max_tokens: int | None = 2000
    timeout_s: int = 120
    retries: int = 3
    retry_backoff_s: float = 5.0

    def __post_init__(self) -> None:
        self._spec = models.get(self.model)

    # ------------------------------------------------------------------
    # text-only chat
    # ------------------------------------------------------------------

    def chat(self, prompt: str, *, system: str | None = "You are an assistant") -> str:
        """Send a one-shot text prompt and return the raw string response."""
        messages: list[dict[str, Any]] = []
        if system and self._spec.supports_system_role:
            messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            # Models like o1-preview reject the 'system' role, so we prepend
            # the instruction to the user message instead.
            merged = f"{system}\n\n{prompt}" if system else prompt
            messages.append({"role": "user", "content": merged})

        body: dict[str, Any] = {
            "model": self._spec.remote_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        return self._post(body)

    # ------------------------------------------------------------------
    # multimodal chat (image input + text output)
    # ------------------------------------------------------------------

    def chat_with_image(self, prompt: str, image_path: str | Path, *,
                        system: str | None = None) -> str:
        """Send a text + image request (used for perception experiments)."""
        if not self._spec.is_multimodal:
            raise ValueError(f"Model {self.model!r} is not multimodal.")

        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        })
        body: dict[str, Any] = {
            "model": self._spec.remote_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return self._post(body)

    def chat_with_image_pair(self, prompt: str,
                             left: str | Path, right: str | Path,
                             *, system: str | None = None) -> str:
        """Send a text + two-image request (pairwise perception comparison)."""
        if not self._spec.is_multimodal:
            raise ValueError(f"Model {self.model!r} is not multimodal.")

        def _b64(p):
            return base64.b64encode(Path(p).read_bytes()).decode()

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_b64(left)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_b64(right)}"}},
            ],
        })
        body: dict[str, Any] = {
            "model": self._spec.remote_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return self._post(body)

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._spec.key_fn()}",
            "Content-Type": "application/json",
        }

    def _post(self, body: dict[str, Any]) -> str:
        """POST to the endpoint and return the primary message content."""
        url = self._spec.endpoint_fn()
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, headers=self._headers(), json=body,
                                  timeout=self.timeout_s)
            except requests.RequestException as e:
                last_exc = e
                log.warning("%s: attempt %d: %s", self.model, attempt, e)
                time.sleep(self.retry_backoff_s * attempt)
                continue

            if r.status_code == 200:
                payload = r.json()
                try:
                    return payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise APIError(
                        f"Unexpected response shape from {self.model}: {payload}"
                    ) from e

            if r.status_code in (429, 500, 502, 503, 504):
                log.warning("%s: attempt %d: status %d: %s",
                            self.model, attempt, r.status_code, r.text[:200])
                time.sleep(self.retry_backoff_s * attempt)
                continue

            raise APIError(
                f"{self.model}: status {r.status_code}: {r.text[:500]}"
            )

        raise APIError(
            f"{self.model}: exhausted {self.retries} retries. Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Image generation clients (Ablai-compatible)
# ---------------------------------------------------------------------------

@dataclass
class ImageGenClient:
    """Client for text-to-image generation (Nano Banana)."""

    model: str = "nano-banana"
    size: str = "1024x1024"
    timeout_s: int = 180
    retries: int = 3
    retry_backoff_s: float = 5.0

    def __post_init__(self) -> None:
        self._spec = models.get(self.model)

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate ``n`` images and return a list of image URLs."""
        url = self._spec.endpoint_fn()
        headers = {"Authorization": f"Bearer {self._spec.key_fn()}"}
        body = {
            "model": self._spec.remote_model,
            "prompt": prompt,
            "n": n,
            "size": self.size,
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, headers=headers, json=body,
                                  timeout=self.timeout_s)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.retry_backoff_s * attempt)
                continue
            if r.status_code == 200:
                payload = r.json()
                return [item.get("url") or item.get("b64_json", "")
                        for item in payload.get("data", [])]
            time.sleep(self.retry_backoff_s * attempt)
        raise APIError(f"Image generation failed after {self.retries} retries: {last_exc}")

    def edit(self, prompt: str, image_path: str | Path, n: int = 1) -> list[str]:
        """Edit an existing image according to ``prompt`` and return URLs."""
        spec = models.get("nano-banana-edit")
        url = spec.endpoint_fn()
        headers = {"Authorization": f"Bearer {spec.key_fn()}"}
        files = {
            "image": open(image_path, "rb"),
            "prompt": (None, prompt),
            "n": (None, str(n)),
            "size": (None, self.size),
            "model": (None, spec.remote_model),
        }
        try:
            r = requests.post(url, headers=headers, files=files, timeout=self.timeout_s)
        finally:
            files["image"].close()
        if r.status_code == 200:
            return [item.get("url") or item.get("b64_json", "")
                    for item in r.json().get("data", [])]
        raise APIError(f"Image edit failed: status {r.status_code}: {r.text[:500]}")
