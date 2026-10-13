"""Structured (key=value / JSON) logging shared by pipeline and API."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} | {record.levelname:<7} | {record.name} | {record.getMessage()}"
        extra = getattr(record, "fields", None)
        if extra:
            base += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
        return base


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        use_json = os.environ.get("MCPGUARD_LOG_JSON", "0") == "1"
        handler.setFormatter(JsonFormatter() if use_json else PrettyFormatter())
        root = logging.getLogger("mcpguard")
        root.handlers[:] = [handler]
        root.setLevel(os.environ.get("MCPGUARD_LOG_LEVEL", "INFO"))
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"mcpguard.{name}")


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
