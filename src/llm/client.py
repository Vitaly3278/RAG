from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def _extract_json_fragment(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_start = stripped.find("```json")
    if fenced_start != -1:
        fenced_end = stripped.find("```", fenced_start + 7)
        if fenced_end != -1:
            candidate = stripped[fenced_start + 7 : fenced_end].strip()
            if candidate:
                return candidate

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and first < last:
        return stripped[first : last + 1]
    return stripped


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
        current_prompt = prompt
        for _ in range(self._retries + 1):
            raw = self.invoke_text(current_prompt, **kwargs)
            try:
                return json.loads(_extract_json_fragment(raw))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                current_prompt = (
                    f"{prompt}\n\nPrevious output was invalid JSON.\n"
                    "Return only one valid JSON object and nothing else."
                )
        raise ValueError(f"LLM did not return valid JSON after retries: {last_error}")
