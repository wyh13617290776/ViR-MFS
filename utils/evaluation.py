"""Evaluation and image-output helpers for ViR-MFS."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import YCrCb2RGB
from utils.metrics import SegmentationMetric


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a normalized tensor as an image file.

    Args:
        tensor: Image tensor with shape ``[1, H, W]`` or ``[3, H, W]``.
        path: Output image path.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255).byte()
    if tensor.shape[0] == 1:
        image = Image.fromarray(tensor[0].numpy())
    else:
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy())
    image.save(path)


@torch.no_grad()
def evaluate_miou(model, data_loader: DataLoader, num_classes: int, device, logger) -> Tuple[list, float]:
    """Evaluate semantic segmentation mIoU.

    Args:
        model: Fusion/segmentation model.
        data_loader: Loader returning either train-style or test-style batches.
        num_classes: Number of semantic classes.
        device: Torch device used for inference.
        logger: Logger used for per-class metric output.

    Returns:
        Tuple ``(per_class_iou, mean_iou)``.
    """
    model.eval()
    metric = SegmentationMetric(num_classes)

    for batch in data_loader:
        vi_image = batch[0].to(device)
        ir_image = batch[1].to(device)
        label_tensor = batch[2].to(device)

        _, seg_logits = model(vi_image, ir_image)
        seg_pred = torch.argmax(seg_logits, dim=1)
        metric.add_batch(label_tensor.cpu().numpy(), seg_pred.cpu().numpy())

    return summarize_iou(metric, num_classes, logger)


def summarize_iou(metric: SegmentationMetric, num_classes: int, logger) -> Tuple[list, float]:
    """Summarize IoU values from a metric accumulator.

    Args:
        metric: Segmentation metric accumulator.
        num_classes: Number of semantic classes.
        logger: Logger used for per-class metric output.

    Returns:
        Tuple ``(per_class_iou, mean_iou)``.
    """
    intersection = np.diag(metric.confusion_matrix)
    union = np.sum(metric.confusion_matrix, axis=1) + np.sum(metric.confusion_matrix, axis=0) - intersection

    per_class_iou = []
    logger.info("[Eval] Per-class IoU:")
    for cls in range(num_classes):
        if union[cls] > 0:
            iou = intersection[cls] / max(1, union[cls])
            per_class_iou.append(iou)
            logger.info(f"  Class {cls:02d}: IoU = {iou:.4f} ({intersection[cls]}/{union[cls]})")
        else:
            per_class_iou.append(np.nan)
            logger.info(f"  Class {cls:02d}: no samples in GT")

    valid_ious = [value for value in per_class_iou if not np.isnan(value)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    logger.info(f"Current Eval mIoU = {mean_iou:.4f}")
    return per_class_iou, mean_iou


@torch.no_grad()
def run_test_inference(
    model,
    data_loader: DataLoader,
    num_classes: int,
    device,
    fusion_save_dir: str,
    seg_save_dir: str,
    logger,
) -> Tuple[list, float]:
    """Run testing inference, save outputs, and compute mIoU.

    Args:
        model: Fusion/segmentation model.
        data_loader: Test data loader.
        num_classes: Number of semantic classes.
        device: Torch device used for inference.
        fusion_save_dir: Directory for fused RGB images.
        seg_save_dir: Directory for predicted segmentation masks.
        logger: Logger used for progress and metrics.

    Returns:
        Tuple ``(per_class_iou, mean_iou)``.
    """
    os.makedirs(fusion_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)
    metric = SegmentationMetric(num_classes)
    model.eval()

    for vi_y, ir_image, label_tensor, name, cb, cr in tqdm(data_loader, total=len(data_loader)):
        vi_y = vi_y.to(device)
        ir_image = ir_image.to(device)
        label_tensor = label_tensor.to(device)
        cb = cb.to(device)
        cr = cr.to(device)

        fused_img, seg_logits = model(vi_y, ir_image)
        seg_pred = torch.argmax(seg_logits, dim=1)
        for item_idx, file_name in enumerate(name):
            # Save each item in the batch so test.batch_size can be configured.
            fused_rgb = YCrCb2RGB(fused_img[item_idx], cb[item_idx], cr[item_idx])
            save_image(fused_rgb, os.path.join(fusion_save_dir, file_name))
            seg_mask = Image.fromarray(seg_pred[item_idx].cpu().numpy().astype(np.uint8))
            seg_mask.save(os.path.join(seg_save_dir, file_name))
        metric.add_batch(label_tensor.cpu().numpy(), seg_pred.cpu().numpy())

    return summarize_iou(metric, num_classes, logger)
