"""Semantic segmentation palette visualization helpers."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
from PIL import Image


def get_msrs_palette() -> np.ndarray:
    """Return the MSRS semantic label palette.

    Args:
        None.

    Returns:
        Palette array with shape ``[9, 3]`` and RGB values.
    """
    return np.array(
        [
            [0, 0, 0],
            [64, 0, 128],
            [64, 64, 0],
            [0, 128, 192],
            [0, 0, 192],
            [128, 128, 0],
            [64, 64, 128],
            [192, 128, 128],
            [192, 64, 0],
        ],
        dtype=np.uint8,
    )


def get_fmb_palette() -> np.ndarray:
    """Return the FMB semantic label palette.

    Args:
        None.

    Returns:
        Palette array with shape ``[15, 3]`` and RGB values.
    """
    return np.array(
        [
            [0, 0, 0],
            [0, 255, 255],
            [255, 105, 180],
            [169, 169, 169],
            [255, 127, 0],
            [255, 69, 0],
            [0, 100, 0],
            [135, 206, 235],
            [0, 0, 255],
            [139, 69, 19],
            [255, 0, 0],
            [255, 0, 255],
            [128, 0, 128],
            [255, 215, 0],
            [50, 205, 50],
        ],
        dtype=np.uint8,
    )


def get_palette(dataset_name: str) -> np.ndarray:
    """Return a semantic visualization palette by dataset name.

    Args:
        dataset_name: Dataset identifier, for example ``MSRS`` or ``FMB``.

    Returns:
        Palette array with shape ``[num_classes, 3]``.

    Raises:
        ValueError: If the dataset has no registered palette.
    """
    palettes: Dict[str, np.ndarray] = {
        "MSRS": get_msrs_palette(),
        "FMB": get_fmb_palette(),
    }
    key = dataset_name.upper()
    if key not in palettes:
        raise ValueError(f"No segmentation palette registered for dataset: {dataset_name}")
    return palettes[key]


def colorize_label(label: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert an integer label map to an RGB visualization.

    Args:
        label: Integer label map with shape ``[H, W]``.
        palette: RGB palette with shape ``[num_classes, 3]``.

    Returns:
        RGB visualization array with shape ``[H, W, 3]``.
    """
    label = np.asarray(label)
    image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    valid_mask = (label >= 0) & (label < len(palette))
    image[valid_mask] = palette[label[valid_mask].astype(np.int64)]
    return image


def save_colorized_label(label: np.ndarray, palette: np.ndarray, path: str) -> None:
    """Save a colorized semantic label map.

    Args:
        label: Integer label map with shape ``[H, W]``.
        palette: RGB palette with shape ``[num_classes, 3]``.
        path: Output image path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(colorize_label(label, palette)).save(path)
