"""Training orchestration for ViR-MFS."""

from __future__ import annotations

import csv
import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config_loader import ConfigInjector
from data_pipeline.dataloader import VIFSDataset
from nets.segformer import SegFormer
from utils.checkpoint import incompatible_keys_to_dict, load_checkpoint, save_checkpoint
from utils.evaluation import evaluate_miou
from utils.experiment import (
    create_run_id,
    prepare_trace_dir,
    runtime_manifest,
    write_json,
    write_yaml,
)
from utils.losses import FusionLoss, ce_loss
from utils.runtime import cleanup_runtime, quiet_logger, setup_runtime
from utils.utils_logger import get_logger
from utils.utils_meta import (
    build_meta_parameter_groups,
    has_invalid_tensor,
    maybe_clip_grad_norm,
    resolve_meta_target,
    run_meta_step,
    split_mtr_mts,
)


def _build_loader(injector: ConfigInjector, split: str, batch_size: int, num_workers: int, resize_size, sampler=None):
    """Build a dataset and dataloader for one split.

    Args:
        injector: Configuration injector.
        split: Dataset split name.
        batch_size: Loader batch size.
        num_workers: Number of dataloader worker processes.
        resize_size: Image resize tuple/list in ``(W, H)`` order.
        sampler: Optional distributed sampler.

    Returns:
        Tuple ``(dataset, dataloader)``.
    """
    dataset = VIFSDataset(
        mode=split,
        resize_size=tuple(resize_size),
        label_resize_interpolation=injector.test_config().get("label_resize_interpolation", "nearest") if split == "test" else injector.train_config().get("label_resize_interpolation", "nearest"),
        **injector.dataset_paths(split),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return dataset, loader


def _write_iou_csv_header(path: str, num_classes: int) -> None:
    """Create the evaluation CSV header when the file is missing.

    Args:
        path: CSV file path.
        num_classes: Number of semantic classes.

    Returns:
        None.
    """
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch"] + [f"class_{idx}_IoU" for idx in range(num_classes)] + ["mIoU"])


def _append_iou_csv(path: str, epoch: int, per_class_iou, mean_iou: float) -> None:
    """Append one evaluation row to the CSV file.

    Args:
        path: CSV file path.
        epoch: One-based epoch index.
        per_class_iou: Per-class IoU sequence.
        mean_iou: Mean IoU value.

    Returns:
        None.
    """
    with open(path, "a", newline="") as stream:
        writer = csv.writer(stream)
        row = [epoch] + [f"{value:.6f}" if not np.isnan(value) else "nan" for value in per_class_iou] + [f"{mean_iou:.6f}"]
        writer.writerow(row)


def _write_loss_csv_header(path: str) -> None:
    """Create the training loss CSV header when the file is missing.

    Args:
        path: CSV file path.

    Returns:
        None.
    """
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "epoch",
            "fusion_loss",
            "seg_loss",
            "valid_batches",
            "skipped_batches",
            "meta_target",
            "meta_fusion_steps",
            "meta_seg_steps",
        ])


def _append_loss_csv(path: str, epoch: int, epoch_summary: dict) -> None:
    """Append one training-loss summary row.

    Args:
        path: CSV file path.
        epoch: One-based epoch index.
        epoch_summary: Dictionary containing epoch-level loss and counter data.

    Returns:
        None.
    """
    with open(path, "a", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            epoch,
            f"{epoch_summary['Lf']:.6f}",
            f"{epoch_summary['Lseg']:.6f}",
            epoch_summary["valid_batches"],
            epoch_summary["skipped_batches"],
            epoch_summary["meta_target"] or "",
            epoch_summary["meta_fusion_steps"],
            epoch_summary["meta_seg_steps"],
        ])


def train(config_path: str = "config/config.yaml", params_path: str = "config/params.yaml") -> None:
    """Run ViR-MFS training or fine-tuning.

    Args:
        config_path: Path to project-level YAML configuration.
        params_path: Path to train/test parameter YAML configuration.

    Returns:
        None.
    """
    injector = ConfigInjector.from_files(config_path=config_path, params_path=params_path)
    train_cfg = injector.train_config()
    context = setup_runtime(seed=train_cfg.get("seed"))
    save_dir = injector.train_save_dir()
    exp_name = injector.exp_name
    run_id = create_run_id("train")
    trace_dir = None

    if context.main_process:
        os.makedirs(save_dir, exist_ok=True)
        trace_dir = prepare_trace_dir(save_dir, run_id, str(injector.project_root))
        logger = get_logger("Train", os.path.join(trace_dir, "train.log"))
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Model: {injector.model_name}")
        logger.info(f"Device: {context.device}")
        if context.distributed:
            logger.info(f"DDP world size: {dist.get_world_size()}")
        shutil.copy(config_path, os.path.join(save_dir, "config_backup.yaml"))
        shutil.copy(params_path, os.path.join(save_dir, "params_backup.yaml"))
        shutil.copy(config_path, os.path.join(trace_dir, "config.yaml"))
        shutil.copy(params_path, os.path.join(trace_dir, "params.yaml"))
        write_yaml(os.path.join(trace_dir, "resolved_config.yaml"), {
            "project": injector.cfg,
            "params": injector.params,
            "train": train_cfg,
            "model": injector.model_config("train"),
            "dataset_paths": {
                "train": injector.dataset_paths("train"),
                "test": injector.dataset_paths("test"),
            },
        })
    else:
        logger = quiet_logger("TrainWorker")

    train_dataset = VIFSDataset(
        mode="train",
        resize_size=tuple(train_cfg["resize_size"]),
        label_resize_interpolation=train_cfg.get("label_resize_interpolation", "nearest"),
        **injector.dataset_paths("train"),
    )
    train_sampler = DistributedSampler(train_dataset) if context.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
    )
    _, test_loader = _build_loader(
        injector,
        split="test",
        batch_size=1,
        num_workers=train_cfg["num_workers"],
        resize_size=train_cfg["resize_size"],
    )

    model = SegFormer(**injector.model_config("train")).to(context.device)
    resume_cfg = train_cfg.get("resume", {})
    resume_report = {
        "enabled": bool(resume_cfg.get("enabled", False)),
        "checkpoint": "",
        "strict": bool(resume_cfg.get("strict", True)),
        "loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if resume_cfg.get("enabled", False):
        checkpoint = injector.checkpoint_path(resume_cfg.get("checkpoint") or None)
        incompatible = load_checkpoint(
            model,
            checkpoint,
            context.device,
            strict=resume_report["strict"],
        )
        resume_report.update(incompatible_keys_to_dict(incompatible))
        resume_report["checkpoint"] = checkpoint
        resume_report["loaded"] = True
        logger.info(f"Loaded resume checkpoint: {checkpoint} | strict={resume_report['strict']}")
        if resume_report["missing_keys"]:
            logger.warning(f"Missing resume keys: {resume_report['missing_keys']}")
        if resume_report["unexpected_keys"]:
            logger.warning(f"Unexpected resume keys: {resume_report['unexpected_keys']}")

    if context.distributed:
        model = DDP(model, device_ids=[context.local_rank], output_device=context.local_rank)

    fusion_names, fusion_params, seg_names, seg_params = build_meta_parameter_groups(
        model,
        distributed=context.distributed,
    )

    opt_fusion = optim.Adam(fusion_params, lr=train_cfg["lr_f"]) if fusion_params else None
    opt_seg = optim.Adam(seg_params, lr=train_cfg["lr_seg"]) if seg_params else None
    opt_all = optim.Adam(model.parameters(), lr=train_cfg["lr_all"])
    scaler = GradScaler(enabled=train_cfg["use_amp"])
    criterion_fusion = FusionLoss(**train_cfg.get("fusion_loss", {})).to(context.device)

    csv_path = os.path.join(save_dir, f"{exp_name}_meta_eval_mIoU.csv")
    loss_csv_path = os.path.join(save_dir, f"{exp_name}_train_loss.csv")
    latest_path = os.path.join(save_dir, f"{exp_name}_meta_latest.pth")
    best_path = os.path.join(save_dir, f"{exp_name}_best_mIoU.pth")
    if context.main_process:
        _write_iou_csv_header(csv_path, train_cfg["num_classes"])
        _write_loss_csv_header(loss_csv_path)
        write_json(os.path.join(trace_dir, "manifest.json"), runtime_manifest(
            "train",
            {
                "run_id": run_id,
                "experiment": exp_name,
                "model_name": injector.model_name,
                "save_dir": save_dir,
                "trace_dir": trace_dir,
                "latest_checkpoint": latest_path,
                "best_checkpoint": best_path,
                "eval_csv": csv_path,
                "loss_csv": loss_csv_path,
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(test_loader.dataset),
                "distributed": context.distributed,
                "local_rank": context.local_rank,
                "fusion_param_count": sum(param.numel() for param in fusion_params),
                "seg_param_count": sum(param.numel() for param in seg_params),
                "total_param_count": sum(param.numel() for param in model.parameters()),
                "resume": resume_report,
            },
        ))

    best_miou = -math.inf
    loss_history = {"Lf": [], "Lseg": []}
    meta_epoch_index = 0

    for epoch in range(train_cfg["epochs"]):
        if context.distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        progress = tqdm(train_loader, total=len(train_loader), ncols=120) if context.main_process else train_loader
        if context.main_process:
            progress.set_description(f"Epoch {epoch}/{train_cfg['epochs'] - 1}")

        epoch_loss = {"Lf": 0.0, "Lseg": 0.0}
        batch_count = 0
        skipped_batches = 0
        meta_fusion_steps = 0
        meta_seg_steps = 0
        meta_target, meta_epoch_index = resolve_meta_target(
            epoch=epoch,
            inner_warmup=int(train_cfg["inner_warmup"]),
            inner_every=int(train_cfg["inner_every"]),
            meta_epoch_index=meta_epoch_index,
        )

        for vi, ir, label in progress:
            vi = vi.to(context.device, non_blocking=True)
            ir = ir.to(context.device, non_blocking=True)
            label = label.to(context.device, non_blocking=True)

            opt_all.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg["use_amp"]):
                fused, seg, _, _ = model(vi, ir, return_lists=True)
                loss_fusion, _, _ = criterion_fusion(fused, vi, ir)
                loss_seg = ce_loss(seg, label)
                loss_total = loss_fusion + loss_seg

            if train_cfg["skip_invalid_loss"] and has_invalid_tensor(loss_total):
                skipped_batches += 1
                continue

            scaler.scale(loss_total).backward()
            if train_cfg.get("grad_clip_norm") is not None:
                scaler.unscale_(opt_all)
                maybe_clip_grad_norm(model.parameters(), train_cfg["grad_clip_norm"])
            scaler.step(opt_all)
            scaler.update()

            meta_info = {}
            if meta_target is not None:
                vi_mtr, vi_mts = split_mtr_mts(vi)
                ir_mtr, ir_mts = split_mtr_mts(ir)
                label_mtr, label_mts = split_mtr_mts(label)

                if vi_mtr.shape[0] > 0 and vi_mts.shape[0] > 0:
                    start = time.time()
                    meta_info["meta_branch"] = meta_target
                    if meta_target == "fusion":
                        stepped = run_meta_step(
                            model=model,
                            params=fusion_params,
                            names=fusion_names,
                            optimizer=opt_fusion,
                            support_loss_fn=lambda outputs: criterion_fusion(outputs[0], vi_mtr, ir_mtr)[0],
                            query_loss_fn=lambda outputs: criterion_fusion(outputs[0], vi_mts, ir_mts)[0],
                            support_args=(vi_mtr, ir_mtr),
                            query_args=(vi_mts, ir_mts),
                            inner_lr=train_cfg["inner_lr"],
                            use_amp=train_cfg["use_amp"],
                            context=context,
                            grad_clip_norm=train_cfg.get("grad_clip_norm"),
                        )
                        meta_info["meta_fusion"] = stepped
                        meta_fusion_steps += int(stepped)
                    else:
                        stepped = run_meta_step(
                            model=model,
                            params=seg_params,
                            names=seg_names,
                            optimizer=opt_seg,
                            support_loss_fn=lambda outputs: ce_loss(outputs[1], label_mtr),
                            query_loss_fn=lambda outputs: ce_loss(outputs[1], label_mts),
                            support_args=(vi_mtr, ir_mtr),
                            query_args=(vi_mts, ir_mts),
                            inner_lr=train_cfg["inner_lr"],
                            use_amp=train_cfg["use_amp"],
                            context=context,
                            grad_clip_norm=train_cfg.get("grad_clip_norm"),
                        )
                        meta_info["meta_seg"] = stepped
                        meta_seg_steps += int(stepped)
                    meta_info["meta_time"] = f"{time.time() - start:.2f}s"

            opt_all.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg["use_amp"]):
                fused2, seg2, _, _ = model(vi, ir, return_lists=True)
                loss_fusion2, _, _ = criterion_fusion(fused2, vi, ir)
                loss_seg2 = ce_loss(seg2, label)
                loss_total2 = loss_fusion2 + loss_seg2

            if not (train_cfg["skip_invalid_loss"] and has_invalid_tensor(loss_total2)):
                scaler.scale(loss_total2).backward()
                if train_cfg.get("grad_clip_norm") is not None:
                    scaler.unscale_(opt_all)
                    maybe_clip_grad_norm(model.parameters(), train_cfg["grad_clip_norm"])
                scaler.step(opt_all)
                scaler.update()
            else:
                skipped_batches += 1

            epoch_loss["Lf"] += float(loss_fusion.detach().item())
            epoch_loss["Lseg"] += float(loss_seg.detach().item())
            batch_count += 1
            if context.main_process:
                progress.set_postfix(Lf=loss_fusion.item(), Lseg=loss_seg.item(), **meta_info)

        if context.main_process:
            epoch_summary = {
                "Lf": epoch_loss["Lf"] / max(1, batch_count),
                "Lseg": epoch_loss["Lseg"] / max(1, batch_count),
                "valid_batches": batch_count,
                "skipped_batches": skipped_batches,
                "meta_target": meta_target,
                "meta_fusion_steps": meta_fusion_steps,
                "meta_seg_steps": meta_seg_steps,
            }
            loss_history["Lf"].append(epoch_summary["Lf"])
            loss_history["Lseg"].append(epoch_summary["Lseg"])
            _append_loss_csv(loss_csv_path, epoch + 1, epoch_summary)
            model_to_eval = model.module if context.distributed else model
            save_checkpoint(model_to_eval, latest_path)

            if (epoch + 1) % int(train_cfg["eval_every"]) == 0:
                per_class_iou, miou = evaluate_miou(
                    model_to_eval,
                    test_loader,
                    train_cfg["num_classes"],
                    context.device,
                    logger,
                    include_absent_classes=bool(train_cfg.get("include_absent_classes_in_miou", False)),
                )
                _append_iou_csv(csv_path, epoch + 1, per_class_iou, miou)
                if miou > best_miou:
                    best_miou = miou
                    save_checkpoint(model_to_eval, best_path)
                    logger.info(f"New best checkpoint: epoch={epoch + 1}, mIoU={miou:.4f}, path={best_path}")
                else:
                    logger.info(f"Eval finished: epoch={epoch + 1}, mIoU={miou:.4f}, best={best_miou:.4f}")

        if context.distributed:
            dist.barrier()

    if context.main_process:
        plt.figure(figsize=(10, 6))
        for key, values in loss_history.items():
            plt.plot(range(1, len(values) + 1), values, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(save_dir, f"{exp_name}_meta_loss_curve.png")
        plt.savefig(fig_path)
        logger.info(f"Training finished. Loss curve saved to {fig_path}")

    cleanup_runtime(context)


if __name__ == "__main__":
    train()
