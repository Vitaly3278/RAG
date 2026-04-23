from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class OllamaBackend:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        temperature = float(kwargs.get("temperature", 0.1))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
            "format": "json",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = json.loads(raw)
        text = parsed.get("response", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned empty response field")
        return text
