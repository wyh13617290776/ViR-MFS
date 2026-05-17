"""Testing orchestration for ViR-MFS."""

from __future__ import annotations

import os
import shutil

import torch
from torch.utils.data import DataLoader

from config_loader import ConfigInjector
from data_pipeline.dataloader import VIFSDataset
from nets.segformer import SegFormer
from utils.checkpoint import incompatible_keys_to_dict, load_checkpoint
from utils.evaluation import run_test_inference
from utils.experiment import (
    create_run_id,
    prepare_trace_dir,
    runtime_manifest,
    write_json,
    write_yaml,
)
from utils.seg_visualization import get_palette
from utils.utils_logger import get_logger


def test_model(config_path: str = "config/config.yaml", params_path: str = "config/params.yaml") -> None:
    """Run model testing and save fused images plus segmentation masks.

    Args:
        config_path: Path to project-level YAML configuration.
        params_path: Path to train/test parameter YAML configuration.

    Returns:
        None.
    """
    injector = ConfigInjector.from_files(config_path=config_path, params_path=params_path)
    test_cfg = injector.test_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_save_dir, seg_save_dir = injector.test_save_dirs()
    os.makedirs(fusion_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)
    visualization_cfg = dict(test_cfg.get("visualization", {}) or {})
    save_pred_color = bool(visualization_cfg.get("save_pred_color", True))
    save_label_color = bool(visualization_cfg.get("save_label_color", True))
    palette_name = str(visualization_cfg.get("palette", "auto"))
    palette_dataset = injector.dataset_name if palette_name.lower() == "auto" else palette_name
    palette = get_palette(palette_dataset) if save_pred_color or save_label_color else None
    pred_color_save_dir = f"{seg_save_dir}_color" if save_pred_color else None
    label_color_save_dir = f"{seg_save_dir}_label_color" if save_label_color else None

    run_id = create_run_id("test")
    trace_dir = prepare_trace_dir(fusion_save_dir, run_id, str(injector.project_root))
    logger = get_logger("Test", os.path.join(trace_dir, "test.log"))
    checkpoint_path = injector.checkpoint_path()
    logger.info("Testing started.")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {injector.dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Fused images: {fusion_save_dir}")
    logger.info(f"Segmentation masks: {seg_save_dir}")
    if pred_color_save_dir:
        logger.info(f"Colorized predicted masks: {pred_color_save_dir}")
    if label_color_save_dir:
        logger.info(f"Colorized ground-truth labels: {label_color_save_dir}")
    shutil.copy(config_path, os.path.join(trace_dir, "config.yaml"))
    shutil.copy(params_path, os.path.join(trace_dir, "params.yaml"))
    write_yaml(os.path.join(trace_dir, "resolved_config.yaml"), {
        "project": injector.cfg,
        "params": injector.params,
        "test": test_cfg,
        "model": injector.model_config("test"),
        "dataset_paths": {
            "test": injector.dataset_paths("test"),
        },
    })

    dataset = VIFSDataset(
        mode="test",
        resize_size=tuple(test_cfg["resize_size"]),
        label_resize_interpolation=test_cfg.get("label_resize_interpolation", "nearest"),
        **injector.dataset_paths("test"),
    )
    loader = DataLoader(
        dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        num_workers=test_cfg["num_workers"],
        pin_memory=True,
    )
    write_json(os.path.join(trace_dir, "manifest.json"), runtime_manifest(
        "test",
        {
            "run_id": run_id,
            "experiment": injector.exp_name,
            "model_name": injector.model_name,
            "checkpoint": checkpoint_path,
            "fusion_save_dir": fusion_save_dir,
            "seg_save_dir": seg_save_dir,
            "pred_color_save_dir": pred_color_save_dir,
            "label_color_save_dir": label_color_save_dir,
            "visualization": {
                **visualization_cfg,
                "resolved_palette": palette_dataset,
            },
            "trace_dir": trace_dir,
            "test_dataset_size": len(dataset),
            "device": str(device),
        },
    ))

    model = SegFormer(**injector.model_config("test")).to(device)
    checkpoint_report = {
        "checkpoint": checkpoint_path,
        "strict": bool(test_cfg.get("checkpoint_strict", False)),
        "loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    try:
        incompatible = load_checkpoint(
            model,
            checkpoint_path,
            device,
            strict=checkpoint_report["strict"],
        )
        checkpoint_report.update(incompatible_keys_to_dict(incompatible))
        checkpoint_report["loaded"] = True
        logger.info(f"Checkpoint loaded successfully. strict={checkpoint_report['strict']}")
        if checkpoint_report["missing_keys"]:
            logger.warning(f"Missing checkpoint keys: {checkpoint_report['missing_keys']}")
        if checkpoint_report["unexpected_keys"]:
            logger.warning(f"Unexpected checkpoint keys: {checkpoint_report['unexpected_keys']}")
    except FileNotFoundError:
        logger.warning(f"Checkpoint not found, using random weights: {checkpoint_path}")
    write_json(os.path.join(trace_dir, "checkpoint_load.json"), checkpoint_report)

    per_class_iou, mean_iou, label_report = run_test_inference(
        model=model,
        data_loader=loader,
        num_classes=test_cfg["num_classes"],
        device=device,
        fusion_save_dir=fusion_save_dir,
        seg_save_dir=seg_save_dir,
        logger=logger,
        include_absent_classes=bool(test_cfg.get("include_absent_classes_in_miou", False)),
        palette=palette,
        pred_color_save_dir=pred_color_save_dir,
        label_color_save_dir=label_color_save_dir,
    )
    write_json(os.path.join(trace_dir, "metrics.json"), {
        "mIoU": mean_iou,
        "per_class_iou": [
            None if value != value else float(value)
            for value in per_class_iou
        ],
        "num_classes": test_cfg["num_classes"],
        "include_absent_classes_in_miou": bool(test_cfg.get("include_absent_classes_in_miou", False)),
        "visualization": {
            **visualization_cfg,
            "resolved_palette": palette_dataset,
            "pred_color_save_dir": pred_color_save_dir,
            "label_color_save_dir": label_color_save_dir,
        },
        "label_report": label_report,
    })
    logger.info(f"Final Mean IoU (mIoU): {mean_iou:.4f}")
    logger.info("Testing finished.")


if __name__ == "__main__":
    test_model()
