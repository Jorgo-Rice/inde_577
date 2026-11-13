"""Minimal structured logging for the reference service."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Serialize standard log records without exposing request payloads."""

    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("request_id", "model_name", "artifact_version", "row_count"):
            if hasattr(record, field):
                document[field] = getattr(record, field)
        return json.dumps(document, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Install one JSON handler on the root logger."""

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
