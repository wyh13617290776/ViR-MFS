"""Checkpoint loading helpers for ViR-MFS."""

from __future__ import annotations

import os
from typing import Mapping

import torch
from torch import nn


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the DDP ``module.`` prefix from checkpoint keys.

    Args:
        state_dict: Raw checkpoint state dictionary.

    Returns:
        A new state dictionary compatible with non-DDP modules.
    """
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_state_dict_file(path: str, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint as a plain state dictionary.

    Args:
        path: Checkpoint file path.
        device: Device used for ``torch.load(map_location=...)``.

    Returns:
        Plain model state dictionary.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If the checkpoint format is unsupported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device | str,
    strict: bool = True,
) -> nn.modules.module._IncompatibleKeys:
    """Load a checkpoint into a model.

    Args:
        model: Target model.
        path: Checkpoint path.
        device: Target runtime device.
        strict: Whether keys must exactly match the model state dict.

    Returns:
        PyTorch incompatible-keys object returned by ``load_state_dict``.
    """
    state_dict = strip_module_prefix(load_state_dict_file(path, device))
    return model.load_state_dict(state_dict, strict=strict)


def incompatible_keys_to_dict(incompatible_keys) -> dict[str, list[str]]:
    """Convert PyTorch incompatible-key output to a serializable dictionary.

    Args:
        incompatible_keys: Object returned by ``nn.Module.load_state_dict``.

    Returns:
        Dictionary with ``missing_keys`` and ``unexpected_keys`` lists.
    """
    return {
        "missing_keys": list(getattr(incompatible_keys, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible_keys, "unexpected_keys", [])),
    }


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save a model state dictionary.

    Args:
        model: Model to save. DDP wrappers are unwrapped automatically.
        path: Output checkpoint path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), path)
