from __future__ import annotations

import hashlib
import math
import re

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """
    Deterministic lightweight embedding for local demos.
    Keeps the project dependency-free for ingestion examples.
    """
    vector = [0.0] * dim
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:4], byteorder="big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
