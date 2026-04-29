"""Experiment traceability helpers for ViR-MFS."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml


def create_run_id(prefix: str) -> str:
    """Create a timestamped run identifier.

    Args:
        prefix: Short run type prefix, for example ``train`` or ``test``.

    Returns:
        Run identifier string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: str | os.PathLike[str]) -> str:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path.

    Returns:
        Directory path as a string.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def write_yaml(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Write a mapping to a YAML file.

    Args:
        path: Output YAML path.
        data: Mapping to serialize.

    Returns:
        None.
    """
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(data), stream, sort_keys=False, allow_unicode=False)


def write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Write a mapping to a JSON file.

    Args:
        path: Output JSON path.
        data: Mapping to serialize.

    Returns:
        None.
    """
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, default=str)


def _run_command(args: list[str], cwd: Optional[str] = None) -> str:
    """Run a command and return captured text.

    Args:
        args: Command argument list.
        cwd: Optional working directory.

    Returns:
        Command stdout and stderr text. Failures are returned as text instead
        of raising so trace capture never blocks training.
    """
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout
    except Exception as exc:
        return f"Failed to run {' '.join(args)}: {exc}\n"


def runtime_manifest(run_type: str, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build a runtime manifest dictionary.

    Args:
        run_type: Run type, for example ``train`` or ``test``.
        extra: Optional extra values to merge into the manifest.

    Returns:
        Runtime manifest dictionary.
    """
    manifest = {
        "run_type": run_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        manifest["cuda_device_count"] = torch.cuda.device_count()
        manifest["cuda_device_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    if extra:
        manifest.update(dict(extra))
    return manifest


def capture_project_state(trace_dir: str, project_root: str) -> None:
    """Capture reproducibility files for a run.

    Args:
        trace_dir: Directory where trace files are written.
        project_root: Repository root.

    Returns:
        None.
    """
    ensure_dir(trace_dir)
    root = Path(project_root)
    for file_name in ("requirements.txt", "run_experiment.sh"):
        source = root / file_name
        if source.exists():
            shutil.copy(source, Path(trace_dir) / file_name)

    (Path(trace_dir) / "git_status.txt").write_text(
        _run_command(["git", "status", "--short"], cwd=project_root),
        encoding="utf-8",
    )
    (Path(trace_dir) / "git_diff.patch").write_text(
        _run_command(["git", "diff"], cwd=project_root),
        encoding="utf-8",
    )
    (Path(trace_dir) / "pip_freeze.txt").write_text(
        _run_command([sys.executable, "-m", "pip", "freeze"], cwd=project_root),
        encoding="utf-8",
    )


def prepare_trace_dir(base_dir: str, run_id: str, project_root: str) -> str:
    """Create a trace directory and capture project state.

    Args:
        base_dir: Base output directory.
        run_id: Run identifier.
        project_root: Repository root.

    Returns:
        Trace directory path.
    """
    trace_dir = ensure_dir(Path(base_dir) / "history" / run_id)
    capture_project_state(trace_dir, project_root)
    return trace_dir
