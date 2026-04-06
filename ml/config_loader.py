"""
Config Loader — Centralized YAML configuration for all ML subsystems.
=====================================================================
Usage:
    from ml.config_loader import load_config, get_training_config, get_serving_config

    cfg = load_config("training")    # loads configs/training.yaml
    cfg = get_training_config()       # cached singleton

All environment variable substitutions (${VAR:-default}) are resolved at load time.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root (assumes ml/ is one level down from project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR:-default} patterns with environment variable values."""
    def _replacer(match):
        var_name = match.group(1)
        default = match.group(3) if match.group(3) else ""
        return os.environ.get(var_name, default)

    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}', _replacer, value)


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve environment variables in a config dict."""
    resolved = {}
    for key, value in d.items():
        if isinstance(value, str):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            resolved[key] = _resolve_dict(value)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_env_vars(v) if isinstance(v, str)
                else _resolve_dict(v) if isinstance(v, dict)
                else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from configs/ directory.

    Args:
        name: Config filename without extension (e.g., "training", "serving")

    Returns:
        Parsed and env-resolved config dict
    """
    yaml_path = _CONFIGS_DIR / f"{name}.yaml"
    yml_path = _CONFIGS_DIR / f"{name}.yml"

    config_path = yaml_path if yaml_path.exists() else yml_path
    if not config_path.exists():
        logger.warning("Config file not found: %s (tried .yaml and .yml)", name)
        return {}

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        resolved = _resolve_dict(raw)
        logger.debug("Loaded config: %s (%d keys)", config_path.name, len(resolved))
        return resolved
    except ImportError:
        logger.warning("PyYAML not installed. Cannot load %s", config_path)
        return {}
    except Exception as e:
        logger.error("Failed to load config %s: %s", config_path, e)
        return {}


@lru_cache(maxsize=1)
def get_training_config() -> dict[str, Any]:
    """Cached training configuration."""
    return load_config("training")


@lru_cache(maxsize=1)
def get_serving_config() -> dict[str, Any]:
    """Cached serving configuration."""
    return load_config("serving")


@lru_cache(maxsize=1)
def get_epsilon_config() -> dict[str, Any]:
    """Cached EPSILON ad engine configuration."""
    return load_config("epsilon")


@lru_cache(maxsize=1)
def get_monitoring_config() -> dict[str, Any]:
    """Cached monitoring configuration."""
    return load_config("monitoring")


@lru_cache(maxsize=1)
def get_spark_config() -> dict[str, Any]:
    """Cached Spark configuration."""
    return load_config("spark")


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get config for a specific model from training.yaml.

    Args:
        model_name: Model key (e.g., "two_tower", "deepfm", "din")

    Returns:
        Model-specific config merged with feature dimensions
    """
    cfg = get_training_config()
    model_cfg = cfg.get(model_name, {})
    features = cfg.get("features", {})
    # Merge features into model config for convenience
    return {**features, **model_cfg}
