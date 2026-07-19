"""Evaluation and image-output helpers for ViR-MFS."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import YCrCb2RGB
from utils.metrics import SegmentationMetric
from utils.seg_visualization import save_colorized_label


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
def evaluate_miou(
    model,
    data_loader: DataLoader,
    num_classes: int,
    device,
    logger,
    include_absent_classes: bool = False,
) -> Tuple[list, float]:
    """Evaluate semantic segmentation mIoU.

    Args:
        model: Fusion/segmentation model.
        data_loader: Loader returning either train-style or test-style batches.
        num_classes: Number of semantic classes.
        device: Torch device used for inference.
        logger: Logger used for per-class metric output.
        include_absent_classes: Whether classes absent from both prediction and
            ground truth should contribute ``0`` to mIoU.

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

    return summarize_iou(metric, num_classes, logger, include_absent_classes)


def summarize_iou(
    metric: SegmentationMetric,
    num_classes: int,
    logger,
    include_absent_classes: bool = False,
) -> Tuple[list, float]:
    """Summarize IoU values from a metric accumulator.

    Args:
        metric: Segmentation metric accumulator.
        num_classes: Number of semantic classes.
        logger: Logger used for per-class metric output.
        include_absent_classes: Whether classes absent from both prediction and
            ground truth should contribute ``0`` to mIoU.

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
            if include_absent_classes:
                per_class_iou.append(0.0)
                logger.info(f"  Class {cls:02d}: IoU = 0.0000 (absent from union)")
            else:
                per_class_iou.append(np.nan)
                logger.info(f"  Class {cls:02d}: no samples in GT")

    valid_ious = [value for value in per_class_iou if not np.isnan(value)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    logger.info(f"Absent classes included in mIoU: {include_absent_classes}")
    logger.info(f"Current Eval mIoU = {mean_iou:.4f}")
    return per_class_iou, mean_iou


def update_label_histogram(histogram: Optional[np.ndarray], label_array: np.ndarray) -> np.ndarray:
    """Accumulate raw ground-truth label id counts.

    Args:
        histogram: Existing histogram or ``None`` for the first batch.
        label_array: Ground-truth label array from one batch.

    Returns:
        Updated label histogram where index equals label id.
    """
    labels = label_array.astype(np.int64).reshape(-1)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64) if histogram is None else histogram

    batch_hist = np.bincount(labels)
    if histogram is None:
        return batch_hist.astype(np.int64)
    if batch_hist.shape[0] > histogram.shape[0]:
        histogram = np.pad(histogram, (0, batch_hist.shape[0] - histogram.shape[0]))
    histogram[:batch_hist.shape[0]] += batch_hist
    return histogram


def summarize_label_distribution(label_histogram: Optional[np.ndarray], num_classes: int, logger) -> dict:
    """Summarize label ids observed during evaluation.

    Args:
        label_histogram: Raw ground-truth label id histogram.
        num_classes: Number of classes configured for the model and metric.
        logger: Logger used for diagnostics.

    Returns:
        Dictionary containing observed, ignored, and absent class ids.
    """
    if label_histogram is None:
        label_histogram = np.zeros(0, dtype=np.int64)

    observed_labels = [int(idx) for idx, count in enumerate(label_histogram) if count > 0]
    ignored_labels = [label for label in observed_labels if label >= num_classes]
    absent_gt_classes = [
        int(cls)
        for cls in range(num_classes)
        if cls >= len(label_histogram) or label_histogram[cls] == 0
    ]
    valid_gt_classes = [cls for cls in range(num_classes) if cls not in absent_gt_classes]

    if ignored_labels:
        logger.warning(
            "Ground-truth label ids %s are >= num_classes=%s and are ignored by IoU. "
            "Increase num_classes or remap ignore labels if these ids are real classes.",
            ignored_labels,
            num_classes,
        )
    if absent_gt_classes:
        logger.info(f"Classes absent from GT under num_classes={num_classes}: {absent_gt_classes}")

    return {
        "observed_labels": observed_labels,
        "ignored_labels": ignored_labels,
        "absent_gt_classes": absent_gt_classes,
        "valid_gt_classes": valid_gt_classes,
        "num_classes": int(num_classes),
        "label_pixel_count": {
            str(idx): int(count)
            for idx, count in enumerate(label_histogram)
            if count > 0
        },
    }


@torch.no_grad()
def run_test_inference(
    model,
    data_loader: DataLoader,
    num_classes: int,
    device,
    fusion_save_dir: str,
    seg_save_dir: str,
    logger,
    include_absent_classes: bool = False,
    palette: Optional[np.ndarray] = None,
    pred_color_save_dir: Optional[str] = None,
    label_color_save_dir: Optional[str] = None,
) -> Tuple[list, float, dict]:
    """Run testing inference, save outputs, and compute mIoU.

    Args:
        model: Fusion/segmentation model.
        data_loader: Test data loader.
        num_classes: Number of semantic classes.
        device: Torch device used for inference.
        fusion_save_dir: Directory for fused RGB images.
        seg_save_dir: Directory for predicted segmentation masks.
        logger: Logger used for progress and metrics.
        include_absent_classes: Whether classes absent from both prediction and
            ground truth should contribute ``0`` to mIoU.
        palette: Optional RGB palette for semantic visualization.
        pred_color_save_dir: Optional directory for colorized predictions.
        label_color_save_dir: Optional directory for colorized ground-truth labels.

    Returns:
        Tuple ``(per_class_iou, mean_iou, label_report)``.
    """
    os.makedirs(fusion_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)
    if pred_color_save_dir is not None:
        os.makedirs(pred_color_save_dir, exist_ok=True)
    if label_color_save_dir is not None:
        os.makedirs(label_color_save_dir, exist_ok=True)
    metric = SegmentationMetric(num_classes)
    label_histogram = None
    model.eval()

    for vi_y, ir_image, label_tensor, name, cb, cr in tqdm(data_loader, total=len(data_loader)):
        vi_y = vi_y.to(device)
        ir_image = ir_image.to(device)
        label_tensor = label_tensor.to(device)
        cb = cb.to(device)
        cr = cr.to(device)

        fused_img, seg_logits = model(vi_y, ir_image)
        seg_pred = torch.argmax(seg_logits, dim=1)
        label_np = label_tensor.cpu().numpy()
        label_histogram = update_label_histogram(label_histogram, label_np)
        for item_idx, file_name in enumerate(name):
            # Save each item in the batch so test.batch_size can be configured.
            fused_rgb = YCrCb2RGB(fused_img[item_idx], cb[item_idx], cr[item_idx])
            save_image(fused_rgb, os.path.join(fusion_save_dir, file_name))
            seg_mask = Image.fromarray(seg_pred[item_idx].cpu().numpy().astype(np.uint8))
            seg_mask.save(os.path.join(seg_save_dir, file_name))
            if palette is not None and pred_color_save_dir is not None:
                save_colorized_label(
                    seg_pred[item_idx].cpu().numpy().astype(np.uint8),
                    palette,
                    os.path.join(pred_color_save_dir, file_name),
                )
            if palette is not None and label_color_save_dir is not None:
                save_colorized_label(
                    label_np[item_idx].astype(np.uint8),
                    palette,
                    os.path.join(label_color_save_dir, file_name),
                )
        metric.add_batch(label_np, seg_pred.cpu().numpy())

    per_class_iou, mean_iou = summarize_iou(
        metric,
        num_classes,
        logger,
        include_absent_classes=include_absent_classes,
    )
    label_report = summarize_label_distribution(label_histogram, num_classes, logger)
    return per_class_iou, mean_iou, label_report
