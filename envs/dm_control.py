"""Utilities for working with MuJoCo Playground DM Control Suite envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ml_collections import ConfigDict
from mujoco_playground import registry


def apply_overrides(
    config: ConfigDict, overrides: Mapping[str, Any], prefix: str = ""
) -> None:
    """Recursively applies overrides to a ConfigDict.

    The overrides structure mirrors the nested shape of the ConfigDict. When a
    leaf value is encountered the function performs an in-place assignment. This
    keeps the MuJoCo Playground config semantics while giving experiments an easy
    way to tweak parameters from simple TOML files.
    """

    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            if key not in config:
                raise KeyError(f"Override path '{path}' not found in config")
            child = config[key]
            if not isinstance(child, ConfigDict):
                raise ValueError(
                    f"Override path '{path}' targets a non-ConfigDict node: {type(child)}"
                )
            apply_overrides(child, value, path)
        else:
            if key not in config:
                raise KeyError(f"Override path '{path}' not found in config")
            config[key] = value


@dataclass
class DMControlEnvConfig:
    """Configuration required to instantiate a DM Control Suite environment."""

    env_name: str
    overrides: Optional[Dict[str, Any]] = None


def load_environment(
    cfg: DMControlEnvConfig,
) -> Tuple[Any, ConfigDict]:
    """Loads a MuJoCo Playground environment and returns it with the config.

    Args:
        cfg: Environment configuration containing the registry name and optional
        overrides that should be applied on top of the playground defaults.

    Returns:
        A tuple of (environment, resolved_config).
    """

    default_cfg = registry.get_default_config(cfg.env_name)
    resolved_cfg = default_cfg.copy_and_resolve_references()

    if cfg.overrides:
        apply_overrides(resolved_cfg, cfg.overrides)

    env = registry.load(cfg.env_name, config=resolved_cfg)
    return env, resolved_cfg


__all__ = ["DMControlEnvConfig", "apply_overrides", "load_environment"]
