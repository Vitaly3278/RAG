from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class LLMClient:
    """
    Minimal JSON-oriented LLM wrapper.
    `backend` can be injected in tests and should return a string.
    """

    def __init__(self, backend: Callable[..., str], retries: int = 2) -> None:
        self._backend = backend
        self._retries = retries

    def invoke_text(self, prompt: str, **kwargs: Any) -> str:
        return self._backend(prompt=prompt, **kwargs)

    def invoke_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self._retries + 1):
            raw = self.invoke_text(prompt, **kwargs)
            try:
                return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ValueError(f"LLM did not return valid JSON after retries: {last_error}")
