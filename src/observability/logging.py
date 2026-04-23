from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "node", "latency_ms", "status", "tokens_in", "tokens_out"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("rag")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def log_node(
    logger: logging.Logger,
    *,
    request_id: str,
    node: str,
    latency_ms: float,
    status: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    logger.info(
        f"node={node} status={status}",
        extra={
            "request_id": request_id,
            "node": node,
            "latency_ms": latency_ms,
            "status": status,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },
    )
