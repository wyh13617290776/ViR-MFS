"""Runtime helpers for devices, DDP, and reproducibility."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime execution context.

    Args:
        device: Torch device used by the current process.
        local_rank: Local DDP rank, or ``-1`` for non-DDP execution.
        distributed: Whether DDP is active.
        main_process: Whether the current process may perform logging and I/O.

    Returns:
        Immutable context object.
    """

    device: torch.device
    local_rank: int
    distributed: bool
    main_process: bool


def setup_runtime(seed: Optional[int] = None) -> RuntimeContext:
    """Initialize random seeds and optional DDP runtime.

    Args:
        seed: Optional random seed applied to Python, NumPy, and Torch.

    Returns:
        Runtime context for the current process.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return RuntimeContext(
        device=device,
        local_rank=local_rank,
        distributed=distributed,
        main_process=(not distributed or local_rank == 0),
    )


def quiet_logger(name: str = "ViR_MFS_quiet") -> logging.Logger:
    """Create a logger that discards messages.

    Args:
        name: Logger name.

    Returns:
        Logger with a null handler.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def cleanup_runtime(context: RuntimeContext) -> None:
    """Tear down distributed runtime when needed.

    Args:
        context: Runtime context returned by ``setup_runtime``.

    Returns:
        None.
    """
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()
