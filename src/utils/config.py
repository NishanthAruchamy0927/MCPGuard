"""Configuration loading.

All pipeline stages read settings from ``configs/config.yaml`` through
:func:`load_config`; paths are resolved relative to the project root so the
code behaves identically from the CLI, tests, the API and Docker.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dict with attribute access and project-root aware path helpers."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve ``paths.<key>`` to an absolute directory (created on demand)."""
        p = PROJECT_ROOT / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def resolve(relative: str | Path) -> Path:
    """Resolve a project-relative path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("MCPGUARD_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
