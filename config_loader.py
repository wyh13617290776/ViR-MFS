"""Configuration loading and dependency injection helpers for ViR-MFS."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import yaml


ConfigDict = Dict[str, Any]


def _read_yaml(path: str | os.PathLike[str]) -> ConfigDict:
    """Load one YAML file.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML content as a dictionary. Empty YAML files return an empty
        dictionary so downstream code can merge them safely.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {yaml_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two dictionaries.

    Args:
        base: Default values.
        override: Values that should overwrite defaults.

    Returns:
        A new dictionary containing the merged configuration.
    """
    merged: ConfigDict = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_by_path(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested value from a dictionary with dot notation.

    Args:
        config: Source configuration dictionary.
        dotted_path: Dot-separated key path, for example ``train.batch_size``.
        default: Value returned when any key in the path is missing.

    Returns:
        The resolved value or ``default``.
    """
    current: Any = config
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_by_path(config: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested dictionary value with dot notation.

    Args:
        config: Mutable configuration dictionary.
        dotted_path: Dot-separated key path.
        value: Value to store at the target path.

    Returns:
        None. The input dictionary is modified in place.
    """
    current: MutableMapping[str, Any] = config
    parts = dotted_path.split(".")
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]
    current[parts[-1]] = value


def load_configs(
    config_path: str = "config/config.yaml",
    params_path: str = "config/params.yaml",
) -> Tuple[ConfigDict, ConfigDict]:
    """Load the project path/config YAML and parameter YAML.

    Args:
        config_path: Path to the environment and path configuration YAML.
        params_path: Path to the training/testing parameter YAML.

    Returns:
        A tuple ``(cfg, params)`` for backward compatibility with older code.
    """
    return _read_yaml(config_path), _read_yaml(params_path)


@dataclass(frozen=True)
class ConfigInjector:
    """Resolve runtime dependencies from YAML configuration.

    Args:
        cfg: Project-level configuration dictionary.
        params: Hyperparameter dictionary.
        project_root: Repository root used to resolve relative paths.

    Returns:
        A lightweight injector object. Methods expose normalized dictionaries
        for datasets, models, training, testing, checkpoints, and output paths.
    """

    cfg: ConfigDict
    params: ConfigDict
    project_root: Path

    @classmethod
    def from_files(
        cls,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml",
    ) -> "ConfigInjector":
        """Create an injector from YAML files.

        Args:
            config_path: Path to the project-level YAML file.
            params_path: Path to the hyperparameter YAML file.

        Returns:
            A configured ``ConfigInjector`` instance.
        """
        cfg, params = load_configs(config_path, params_path)
        return cls(cfg=cfg, params=params, project_root=Path.cwd())

    def resolve_path(self, path_value: str | os.PathLike[str]) -> str:
        """Resolve a path relative to the project root when needed.

        Args:
            path_value: Absolute or relative path from the YAML config.

        Returns:
            Absolute path string.
        """
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return str(path)
        return str((self.project_root / path).resolve())

    @property
    def model_name(self) -> str:
        """Return the configured model architecture name.

        Args:
            None.

        Returns:
            Model name used for output directory grouping.
        """
        return str(self.cfg.get("model_name", self.cfg.get("model", {}).get("name", "SegFormer")))

    @property
    def dataset_name(self) -> str:
        """Return the active dataset name.

        Args:
            None.

        Returns:
            Dataset identifier, for example ``MSRS`` or ``FMB``.
        """
        return str(self.cfg["dataset"]["name"])

    @property
    def exp_name(self) -> str:
        """Build the canonical experiment name.

        Args:
            None.

        Returns:
            Experiment name in ``{dataset}_{backbone_phi}`` format.
        """
        return f"{self.dataset_name}_{self.backbone_config()['phi']}"

    def dataset_paths(self, split: str) -> ConfigDict:
        """Build visible, infrared, and label paths for one split.

        Args:
            split: Dataset split name, usually ``train`` or ``test``.

        Returns:
            Dictionary with ``vi_dir``, ``ir_dir``, and ``label_dir``.
        """
        root_dir = self.resolve_path(self.cfg["dataset"]["root_dir"])
        base_path = Path(root_dir) / self.dataset_name
        return {
            "vi_dir": str(base_path / "vi" / split),
            "ir_dir": str(base_path / "ir" / split),
            "label_dir": str(base_path / "label" / split),
        }

    def backbone_config(self) -> ConfigDict:
        """Return normalized backbone configuration.

        Args:
            None.

        Returns:
            Dictionary with backbone variant and pretrained directory.
        """
        backbone = dict(self.cfg.get("backbone", {}))
        backbone.setdefault("phi", "b0")
        backbone.setdefault("pretrained_dir", "model_data")
        backbone["pretrained_dir"] = self.resolve_path(backbone["pretrained_dir"])
        return backbone

    def model_config(self, mode: str) -> ConfigDict:
        """Return model constructor parameters for a runtime mode.

        Args:
            mode: Runtime mode, either ``train`` or ``test``.

        Returns:
            Dictionary passed to ``SegFormer``.
        """
        section = self.params.get(mode, {})
        model_params = deep_merge(self.params.get("model", {}), section.get("model", {}))
        wavelet_params = deep_merge(
            self.params.get("wavelet", {}),
            section.get("wavelet", {}),
        )
        backbone = self.backbone_config()
        return {
            "num_classes": int(section.get("num_classes", self.params.get("num_classes", 9))),
            "pretrained": bool(section.get("use_pretrained", False)),
            "backbone_phi": backbone["phi"],
            "pretrained_dir": backbone["pretrained_dir"],
            "wavelet_config": wavelet_params,
            **model_params,
        }

    def train_config(self) -> ConfigDict:
        """Return normalized training parameters.

        Args:
            None.

        Returns:
            Training parameter dictionary.
        """
        train_cfg = dict(self.params.get("train", {}))
        train_cfg.setdefault("batch_size", 4)
        train_cfg.setdefault("num_workers", 4)
        train_cfg.setdefault("epochs", 200)
        train_cfg.setdefault("resize_size", [640, 480])
        train_cfg.setdefault("use_amp", True)
        train_cfg.setdefault("lr_f", 5.0e-5)
        train_cfg.setdefault("lr_seg", 5.0e-5)
        train_cfg.setdefault("lr_all", 5.0e-4)
        train_cfg.setdefault("inner_lr", 1.0e-5)
        train_cfg.setdefault("inner_every", 3)
        train_cfg.setdefault("inner_warmup", 1)
        train_cfg.setdefault("grad_clip_norm", None)
        train_cfg.setdefault("skip_invalid_loss", True)
        train_cfg.setdefault("eval_every", 1)
        train_cfg.setdefault("resume", {})
        return train_cfg

    def test_config(self) -> ConfigDict:
        """Return normalized testing parameters.

        Args:
            None.

        Returns:
            Testing parameter dictionary.
        """
        test_cfg = dict(self.params.get("test", {}))
        test_cfg.setdefault("batch_size", 1)
        test_cfg.setdefault("num_workers", 4)
        test_cfg.setdefault("resize_size", [640, 480])
        test_cfg.setdefault("num_classes", self.train_config().get("num_classes", 9))
        test_cfg.setdefault("checkpoint_strict", False)
        return test_cfg

    def train_save_dir(self) -> str:
        """Return the training output directory.

        Args:
            None.

        Returns:
            Absolute directory path for checkpoints and logs.
        """
        base_dir = self.resolve_path(self.cfg.get("train", {}).get("save_base_dir", "runs_meta"))
        return str(Path(base_dir) / self.model_name / self.exp_name)

    def test_save_dirs(self) -> Tuple[str, str]:
        """Return output directories for fused images and segmentation masks.

        Args:
            None.

        Returns:
            Tuple ``(fusion_dir, segmentation_dir)``.
        """
        base_dir = self.resolve_path(self.cfg.get("test", {}).get("save_base_dir", "test_results"))
        fusion_dir = Path(base_dir) / self.model_name / f"{self.exp_name}_results"
        return str(fusion_dir), f"{fusion_dir}_seg"

    def checkpoint_path(self, name: Optional[str] = None) -> str:
        """Resolve a checkpoint path.

        Args:
            name: Optional checkpoint file name or absolute path. If omitted,
                ``cfg.test.checkpoint_name`` is used.

        Returns:
            Absolute checkpoint path.
        """
        target = name or self.cfg.get("test", {}).get("checkpoint_name", "meta_latest")
        target_path = Path(str(target)).expanduser()
        if target_path.is_absolute():
            return str(target_path)

        file_name = str(target)
        if not file_name.endswith(".pth"):
            file_name = f"{self.exp_name}_{file_name}.pth"
        return str(Path(self.train_save_dir()) / file_name)
