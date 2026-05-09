"""Config loader with ${VAR:default} environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "app_config.yaml"


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:default} patterns in a string."""
    def _replace(match: re.Match) -> str:
        spec = match.group(1)
        if ":" in spec:
            var, default = spec.split(":", 1)
            return os.environ.get(var.strip(), default.strip())
        return os.environ.get(spec.strip(), match.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _walk_expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(i) for i in obj]
    return obj


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load app_config.yaml with env-var expansion.

    Args:
        path: Path to YAML config file. Defaults to configs/app_config.yaml.

    Returns:
        Fully expanded config dict.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = path or _CONFIG_PATH
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    _CONFIG = _walk_expand(raw)
    return _CONFIG


def get(key_path: str, default: Any = None) -> Any:
    """Get a nested config value by dot-separated key path.

    Args:
        key_path: Dot-separated path, e.g. ``"llm.model"``.
        default: Returned if the key is not found.

    Returns:
        Config value or default.
    """
    cfg = load_config()
    parts = key_path.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
