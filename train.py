# 支持多卡训练
import os
import time
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.stateless import functional_call # torch 1.x用法
# from torch.func import functional_call # torch 2.x新用法
from tqdm import tqdm

from data_pipeline.dataloader import VIFSDataset
from nets.segformer import SegFormer
from utils.utils_meta import (
    get_fusion_param_names, make_params_dict, split_mtr_mts
)
from utils.losses import FusionLoss, ce_loss
from utils.metrics import SegmentationMetric
from utils.utils_logger import get_logger
from config_loader import load_configs

@torch.no_grad()
def evaluate_miou(model, test_loader, num_classes, device, logger):
    """
    在测试集上评估分割网络的 mIoU (Mean Intersection over Union)。
    """
    model.eval()
    metric = SegmentationMetric(num_classes)

    for batch in test_loader:
        # 测试集 dataloader 返回: vi_y, ir, label, name, cr, cb
        vi_image, ir_image, label_tensor = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        _, seg_logits = model(vi_image, ir_image)
        seg_pred = torch.argmax(seg_logits, dim=1)  # 获取预测类别 [B,H,W]

        pred_np  = seg_pred.cpu().numpy()
        label_np = label_tensor.cpu().numpy()

        # 直接通过混淆矩阵累加一个 batch 的结果，速度极快
        metric.add_batch(label_np, pred_np)

    # 提取交集和并集，无缝衔接原有的日志输出
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

    valid_ious = [v for v in per_class_iou if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if len(valid_ious) else 0.0
    logger.info(f"Current Eval mIoU = {miou:.4f}")
    return per_class_iou, miou

def train():
    cfg, params = load_configs()
    dataset_cfg = cfg['dataset']
    t_params = params['train']
    model_name = cfg.get('model_name')
    
    # === [新增] DDP 环境初始化 ===
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 1. 动态构建保存路径并备份配置 ===
    exp_name = f"{dataset_cfg['name']}_{cfg['backbone']['phi']}"
    save_dir = os.path.join(cfg['train']['save_base_dir'], model_name, exp_name)
    
    # === 初始化 Logger (DDP限制：仅在 0号进程/主卡 上执行 I/O 操作) ===
    if not is_distributed or local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file_path = os.path.join(save_dir, f"{exp_name}_train.log")
        logger = get_logger(name="Train", log_file=log_file_path)
        logger.info(f"================ 开始训练工程: {exp_name} ================")
        if is_distributed:
            logger.info(f"已开启 DDP 分布式训练，总显卡数: {dist.get_world_size()}")
        
        shutil.copy("config/config.yaml", os.path.join(save_dir, "config_backup.yaml"))
        shutil.copy("config/params.yaml", os.path.join(save_dir, "params_backup.yaml"))
    else:
        # 为其他子进程创建一个静默的 logger，防止后续 logger.info 报错
        logger = logging.getLogger("dummy")
        logger.addHandler(logging.NullHandler())

    # === 2. 构建数据加载器 (依赖注入) ===
    def get_dataset_paths(mode):
        base_path = os.path.join(dataset_cfg['root_dir'], dataset_cfg['name'])
        return {
            'vi_dir': os.path.join(base_path, 'vi', mode),
            'ir_dir': os.path.join(base_path, 'ir', mode),
            'label_dir': os.path.join(base_path, 'label', mode),
        }

    train_paths = get_dataset_paths('train')
    train_dataset = VIFSDataset(mode='train', resize_size=tuple(t_params['resize_size']), **train_paths)
    
    # [新增] 为训练集添加分布式采样器
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=t_params['batch_size'], shuffle=(train_sampler is None),
        num_workers=t_params['num_workers'], pin_memory=True, sampler=train_sampler
    )
    
    test_paths = get_dataset_paths('test')
    test_dataset = VIFSDataset(mode='test', resize_size=tuple(t_params['resize_size']), **test_paths)
    # 测试集为了保持代码最小改动和指标完整性，强制只在主卡测全部数据，不使用分布式采样器
    test_loader  = DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        num_workers=t_params['num_workers'], pin_memory=True
    )

    # === 3. 初始化模型与优化器 ===
    model = SegFormer(num_classes=t_params['num_classes'], pretrained=t_params['use_pretrained']).to(device)
    
    # [新增] 包装 DDP 模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 提取融合头与分割头参数 (兼容 DDP 的 "module." 前缀)
    fusion_prefixes = ("module.f0", "module.f1", "module.f2", "module.f3", "module.fusion_head") if is_distributed else ("f0", "f1", "f2", "f3", "fusion_head")
    all_named = list(model.named_parameters())
    
    F_pairs = [(n, p) for (n, p) in all_named if any(n.startswith(k) for k in fusion_prefixes) and p.requires_grad]
    F_names, F_params = zip(*F_pairs) if len(F_pairs) > 0 else ([], [])

    seg_pairs = [(n, p) for (n, p) in all_named if "decode_head" in n]
    S_names, S_params = zip(*seg_pairs) if len(seg_pairs) > 0 else ([], [])

    opt_F   = optim.Adam(F_params, lr=t_params['lr_f']) if len(F_params) > 0 else None
    opt_S   = optim.Adam(S_params, lr=t_params['lr_seg']) if len(S_params) > 0 else None
    opt_all = optim.Adam(model.parameters(), lr=t_params['lr_all'])

    scaler = GradScaler(enabled=t_params['use_amp'])

    # === 4. 日志准备 ===
    loss_history = {"Lf": [], "Lseg": []}
    csv_path = os.path.join(save_dir, f"{exp_name}_meta_eval_mIoU.csv")
    best_ckpt_path = os.path.join(save_dir, f"{exp_name}_best_mIoU.pth")
    
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch"] + [f"class_{i}_IoU" for i in range(t_params['num_classes'])] + ["mIoU"]
            writer.writerow(header)
    
    best_miou = -1.0

    # === 5. 训练循环 ===
    for epoch in range(t_params['epochs']):
        # [新增] 必须让 sampler 知道当前 epoch，以保证数据打乱
        if is_distributed:
            train_sampler.set_epoch(epoch)

        criterion_fusion = FusionLoss().to(device)

        model.train()
        
        # [新增] 仅在主进程打印进度条
        if not is_distributed or local_rank == 0:
            pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
            pbar.set_description(f"Epoch {epoch}/{t_params['epochs']-1}")
        else:
            pbar = train_loader

        epoch_loss = {"Lf": 0.0, "Lseg": 0.0}
        count = 0

        for vi, ir, lbl in pbar:
            vi  = vi.to(device, non_blocking=True)
            ir  = ir.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            # ---------------------------------------------------------
            # 步骤 0: 快速优化整体网络 (一次前向+反向传播)
            # ---------------------------------------------------------
            opt_all.zero_grad(set_to_none=True)
            with autocast(enabled=t_params['use_amp']):
                fused, seg, fu_list, fe_list = model(vi, ir, return_lists=True)
                Lf, _, _ = criterion_fusion(fused, vi, ir)
                loss_seg = ce_loss(seg, lbl)
                L_total = Lf + loss_seg
            scaler.scale(L_total).backward()
            scaler.step(opt_all)
            scaler.update()

            L_out = Lf # 用于进度条显示

            # ---------------------------------------------------------
            # 步骤 1: 融合头 (Fusion Head) 的二阶 Meta 更新 (MAML变体)
            # 核心机制: 在 Support set (mtr) 算一阶导并"模拟更新"参数，
            # 然后用模拟更新后的参数在 Query set (mts) 上算损失，并求关于原参数的二阶导。
            # ---------------------------------------------------------
            if (epoch >= t_params['inner_warmup']) and ((epoch % t_params['inner_every']) == 0) and (len(F_params) > 0) and (epoch % 2 == 0):
                vi_mtr, vi_mts = split_mtr_mts(vi)
                ir_mtr, ir_mts = split_mtr_mts(ir)
                
                if vi_mtr.shape[0] > 0 and vi_mts.shape[0] > 0:
                    t0 = time.time()
                    
                    # [内循环 Inner Loop]: 使用 Support Set 计算一阶梯度
                    with autocast(enabled=t_params['use_amp']):
                        fused_mtr, _, fu_mtr, fe_mtr = model(vi_mtr, ir_mtr, return_lists=True)
                        Lf_1, _, _ = criterion_fusion(fused_mtr, vi_mtr, ir_mtr)

                    # retain_graph=True 和 create_graph=True 是计算二阶导的关键，保留计算图用于后续反传
                    grads_F = torch.autograd.grad(Lf_1, F_params, create_graph=True, retain_graph=True, allow_unused=True)
                    grads_F = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_F, F_params)]

                    # 利用获取的梯度构造"虚拟参数" updated，原模型权重保持不动
                    all_params = make_params_dict(model)
                    updated = dict(all_params)
                    for n, p, g in zip(F_names, F_params, grads_F):
                        updated[n] = p - t_params['inner_lr'] * g

                    # [外循环 Outer Loop]: 使用 Query Set 和"虚拟参数"重新前向传播
                    with autocast(enabled=t_params['use_amp']):
                        fused_mts, _, *_ = functional_call(
                            model, updated, args=(vi_mts, ir_mts), kwargs={"return_lists": True}
                        )
                        if isinstance(fused_mts, tuple): fused_mts = fused_mts[0]
                        Lf_mts, _, _ = criterion_fusion(fused_mts, vi_mts, ir_mts)

                    # 计算外循环损失对【原始参数】的二阶梯度
                    grads_meta_F = torch.autograd.grad(Lf_mts, F_params, retain_graph=False, allow_unused=True)
                    grads_meta_F = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_meta_F, F_params)]

                    # 利用优化器实际更新融合头的参数
                    if opt_F is not None:
                        opt_F.zero_grad(set_to_none=True)
                        for p, g in zip(F_params, grads_meta_F):
                            p.grad = g.detach().clone()
                            
                        # [DDP 梯度同步逻辑]：手动聚合所有显卡的 Meta 梯度
                        if is_distributed:
                            for p in F_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                                    p.grad.data /= dist.get_world_size()

                        opt_F.step()

                    inner_time = time.time() - t0
                else:
                    inner_time = 0.0
                if not is_distributed or local_rank == 0:
                    pbar.set_postfix(inner_time=f"{inner_time:.2f}s", Lf=Lf.item(), Lseg=loss_seg.item())
            else:
                if not is_distributed or local_rank == 0:
                    pbar.set_postfix(Lf=Lf.item(), Lseg=loss_seg.item())

            # ---------------------------------------------------------
            # 步骤 2: 再次快速优化整体网络
            # ---------------------------------------------------------
            opt_all.zero_grad(set_to_none=True)
            with autocast(enabled=t_params['use_amp']):
                fused2, seg2, _, _ = model(vi, ir, return_lists=True)
                Lf2, _, _ = criterion_fusion(fused2, vi, ir)
                loss_seg2 = ce_loss(seg2, lbl)
                L_total2 = Lf2 + loss_seg2
            scaler.scale(L_total2).backward()
            scaler.step(opt_all)
            scaler.update()
            
            # ---------------------------------------------------------
            # 步骤 3: 分割头 (Seg Head) 的二阶 Meta 更新
            # 逻辑同上，只是作用目标从 Fusion Head 换成了 Seg Head
            # ---------------------------------------------------------
            if (epoch >= t_params['inner_warmup']) and ((epoch % t_params['inner_every']) == 0) and (len(S_params) > 0) and (epoch % 2 != 0):
                vi_mtr, vi_mts = split_mtr_mts(vi)
                ir_mtr, ir_mts = split_mtr_mts(ir)
                try:
                    lbl_mtr, lbl_mts = split_mtr_mts(lbl)
                except Exception:
                    lbl_mtr = lbl_mts = None

                if (lbl_mtr is not None) and (vi_mtr.shape[0] > 0) and (vi_mts.shape[0] > 0):
                    t0 = time.time()
                    
                    with autocast(enabled=t_params['use_amp']):
                        _, seg_mtr, *_ = model(vi_mtr, ir_mtr, return_lists=True)
                        Lseg_mtr = ce_loss(seg_mtr, lbl_mtr)

                    grads_S = torch.autograd.grad(Lseg_mtr, S_params, create_graph=True, retain_graph=True, allow_unused=True)
                    grads_S = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_S, S_params)]

                    all_params = make_params_dict(model)
                    updated = dict(all_params)
                    for n, p, g in zip(S_names, S_params, grads_S):
                        updated[n] = p - t_params['inner_lr'] * g

                    with autocast(enabled=t_params['use_amp']):
                        _, seg_mts, *_ = functional_call(
                            model, updated, args=(vi_mts, ir_mts), kwargs={"return_lists": True}
                        )
                        Lseg_mts = ce_loss(seg_mts, lbl_mts)

                    grads_meta_S = torch.autograd.grad(Lseg_mts, S_params, retain_graph=False, allow_unused=True)
                    grads_meta_S = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_meta_S, S_params)]

                    if opt_S is not None:
                        opt_S.zero_grad(set_to_none=True)
                        for p, g in zip(S_params, grads_meta_S):
                            p.grad = g.detach().clone()
                            
                        # [DDP 梯度同步逻辑]：手动聚合所有显卡的 Meta 梯度
                        if is_distributed:
                            for p in S_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                                    p.grad.data /= dist.get_world_size()

                        opt_S.step()

                    inner_time_seg = time.time() - t0
                else:
                    inner_time_seg = 0.0
                if not is_distributed or local_rank == 0:
                    pbar.set_postfix(inner_time_seg=f"{inner_time_seg:.2f}s", Lf=Lf.item(), Lseg=loss_seg.item())

            epoch_loss["Lf"]   += Lf.item()
            epoch_loss["Lseg"] += loss_seg.item()
            count += 1

        # --- 记录损失并进行评估 (DDP限制：仅在主进程进行测试和保存) ---
        if not is_distributed or local_rank == 0:
            loss_history["Lf"].append(epoch_loss["Lf"] / max(1, count))
            loss_history["Lseg"].append(epoch_loss["Lseg"] / max(1, count))

            # [重要] 剥离 DDP 外壳后再保存权重
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, f"{exp_name}_meta_latest.pth"))
            
            per_cls_iou, miou = evaluate_miou(model_to_save, test_loader, t_params['num_classes'], device, logger)
            
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = [epoch + 1] + [f"{x:.6f}" if not np.isnan(x) else "nan" for x in per_cls_iou] + [f"{miou:.6f}"]
                writer.writerow(row)
            
            if miou > best_miou:
                best_miou = miou
                torch.save(model_to_save.state_dict(), best_ckpt_path)
                logger.info(f"发现最佳模型！Epoch {epoch+1}: mIoU={miou:.4f} | 已保存至 {best_ckpt_path}")
            else:
                logger.info(f"[Eval] Epoch {epoch+1} 评估结束 | 当前 mIoU={miou:.4f} | 历史最佳 mIoU={best_miou:.4f}")

        # [新增] 设置屏障，让子进程在此等待，直到主进程完成评估与保存
        if is_distributed:
            dist.barrier()

    # --- 绘制曲线 (仅主进程) ---
    if not is_distributed or local_rank == 0:
        plt.figure(figsize=(10,6))
        for k, v in loss_history.items():
            plt.plot(range(1, t_params['epochs']+1), v, label=k)
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training Loss Curves"); plt.legend(); plt.grid(True)
        fig_path = os.path.join(save_dir, f"{exp_name}_meta_loss_curve.png")
        plt.savefig(fig_path)
        logger.info(f"================ 训练全部完成！Loss 曲线已保存到 {fig_path} ================")

if __name__ == "__main__":
    train()