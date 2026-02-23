# new_train_meta_v2.py —— 单卡 + 元学习（包含对融合 head / 分割 head 的二阶更新）
import os
import time
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch.amp import autocast, GradScaler
from torch.func import functional_call
from tqdm import tqdm

from dataloder import vifs_dataloder
from model_lite.nets.segformer import SegFormer
from utils_meta import (
    load_partial_weights, get_fusion_param_names, make_params_dict,
    merge_updated_params, fusion_loss, ce_loss, split_mtr_mts
)

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- 数据集 ----------------
    train_dataset = vifs_dataloder()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # ---------------- 模型（单卡） ----------------
    try:
        model = SegFormer(num_classes=args.num_classes).cuda()
    except TypeError:
        print(f"[INFO] Fusion_Seg_Model 不支持 num_classes，已忽略 --num_classes={args.num_classes}")

    # model = load_partial_weights(model, args.ckpt_path, device="cuda")
    m = model  # 简写

    # 参数分组
    fusion_param_names = set(get_fusion_param_names(m))
    all_named = list(m.named_parameters())
    F_pairs = [(n, p) for (n, p) in all_named if n in fusion_param_names and p.requires_grad]
    F_names, F_params = zip(*F_pairs) if len(F_pairs) > 0 else ([], [])

    seg_pairs = [(n, p) for (n, p) in all_named if n.startswith("decode_head") or "decode_head" in n]
    S_names, S_params = zip(*seg_pairs) if len(seg_pairs) > 0 else ([], [])

    # mfe_params    = list(m.mfe.parameters())

    opt_F   = optim.Adam(F_params, lr=args.lr_f) if len(F_params) > 0 else None
    opt_S   = optim.Adam(S_params, lr=args.lr_seg) if len(S_params) > 0 else None
    # opt_MFE = optim.Adam(mfe_params,    lr=args.lr_mfe)

    # 整体快速优化器
    opt_all = optim.Adam(m.parameters(), lr=args.lr_all)

    scaler = GradScaler('cuda', enabled=args.use_amp)

    # 记录 loss 曲线
    loss_history = {"Lf": [], "Lg": [], "Lseg": [], "Lout": []}

    # ---------------- 训练循环 ----------------
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
        pbar.set_description(f"Epoch {epoch}/{args.epochs-1}")

        epoch_loss = {"Lf": 0.0, "Lg": 0.0, "Lseg": 0.0, "Lout": 0.0}
        count = 0

        for vi, ir, lbl, vi_gt in pbar:
            vi   = vi.cuda(non_blocking=True)
            ir   = ir.cuda(non_blocking=True)
            lbl  = lbl.cuda(non_blocking=True)

            # ------------------ 0) 快速优化整体网络（一次） ------------------
            opt_all.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=args.use_amp):
                fused, seg, fu_list, fe_list = model(vi, ir, return_lists=True)
                # fe_list_detached = [f.detach() for f in fe_list]
                # Lg, _, _ = model.mfe(fu_list, fe_list_detached)
                Lf, _, _ = fusion_loss(fused, vi, ir)
                loss_seg = ce_loss(seg, lbl)
                L_total = Lf + loss_seg
            scaler.scale(L_total).backward()
            scaler.step(opt_all)
            scaler.update()

            # ---- 为外部统计保留最新值（供显示） ----
            L_out = Lf

            # ------------------ 1) 融合 head 的二阶 Meta 更新 ------------------
            # 仅在满足内更新触发条件下，并且存在可用的 F_params
            if (epoch >= args.inner_warmup) and ((epoch % args.inner_every) == 0) and (len(F_params) > 0) and (epoch%2==0):
                # 划分 mtr/mts（同样对 lbl 做分割以备后用）
                vi_mtr, vi_mts = split_mtr_mts(vi)
                ir_mtr, ir_mts = split_mtr_mts(ir)
                try:
                    lbl_mtr, lbl_mts = split_mtr_mts(lbl)
                except Exception:
                    lbl_mtr = lbl_mts = None

                if vi_mtr.shape[0] > 0 and vi_mts.shape[0] > 0:
                    t0 = time.time()
                    # 在 mtr 上计算 Lg_mtr（MFE 的任务 loss，fe 不反传）
                    with autocast('cuda', enabled=args.use_amp):
                        fused_mtr, _, fu_mtr, fe_mtr = model(vi_mtr, ir_mtr, return_lists=True)
                        # fe_mtr_detached = [f.detach() for f in fe_mtr]
                        # Lg_mtr, _, _ = model.mfe(fu_mtr, fe_mtr_detached)
                        Lf_1, _, _ = fusion_loss(fused_mtr, vi_mtr, ir_mtr)

                    # 1) 对 F_params 求一阶梯度（作为“内步”）
                    grads_F = torch.autograd.grad(
                        Lf_1, F_params,
                        create_graph=True, retain_graph=True, allow_unused=True
                    )
                    grads_F = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_F, F_params)]

                    # 2) 用一阶梯度构造更新后的 F' 参数字典
                    all_params = make_params_dict(m)   # name -> tensor (当前)
                    updated = dict(all_params)
                    for n, p, g in zip(F_names, F_params, grads_F):
                        updated[n] = p - args.inner_lr * g

                    # 3) 在 mts 上用 F' 前向，得到 Lf_mts（融合质量 loss）
                    with autocast('cuda', enabled=args.use_amp):
                        fused_mts, _, *_ = functional_call(
                            model, updated, args=(vi_mts, ir_mts), kwargs={"return_lists": True}
                        )
                        if isinstance(fused_mts, tuple):
                            fused_mts = fused_mts[0]
                        Lf_mts, _, _ = fusion_loss(fused_mts, vi_mts, ir_mts)

                    # 4) 计算 Lf_mts 对原始 F_params 的梯度（这就是二阶项），并用 opt_F 更新 F_params
                    grads_meta_F = torch.autograd.grad(
                        Lf_mts, F_params,
                        retain_graph=False, allow_unused=True
                    )
                    grads_meta_F = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_meta_F, F_params)]

                    # 将二阶梯度放入对应参数 .grad 并 step opt_F（只更新 fusion）
                    if opt_F is not None:
                        opt_F.zero_grad(set_to_none=True)
                        for p, g in zip(F_params, grads_meta_F):
                            p.grad = g.detach().clone()
                        opt_F.step()

                    t1 = time.time()
                    inner_time = t1 - t0
                else:
                    inner_time = 0.0
                # 在进度条显示
                pbar.set_postfix(inner_time=f"{inner_time:.2f}s", Lf=Lf.item(), Lseg=loss_seg.item(), Lout=L_out.item())
            else:
                pbar.set_postfix(Lf=Lf.item(), Lseg=loss_seg.item(), Lout=L_out.item())

            # ------------------ 2) 再次快速优化整体网络（第二次快速步） ------------------
            opt_all.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=args.use_amp):
                fused2, seg2, fu_list2, fe_list2 = model(vi, ir, return_lists=True)
                fe_list2_detached = [f.detach() for f in fe_list2]
                # Lg2, _, _ = model.mfe(fu_list2, fe_list2_detached)
                Lf2, _, _ = fusion_loss(fused2, vi, ir)
                loss_seg2 = ce_loss(seg2, lbl)
                L_total2 = Lf2 + loss_seg2
            scaler.scale(L_total2).backward()
            scaler.step(opt_all)
            scaler.update()

            # ------------------ 3) 分割 head 的二阶 Meta 更新 ------------------
            # 要求：存在可分割的 lbl，并且存在 S_params
            if (epoch >= args.inner_warmup) and ((epoch % args.inner_every) == 0) and (len(S_params) > 0) and (epoch%2!=0):
                # 再次划分（与上面保持独立）
                vi_mtr, vi_mts = split_mtr_mts(vi)
                ir_mtr, ir_mts = split_mtr_mts(ir)
                try:
                    lbl_mtr, lbl_mts = split_mtr_mts(lbl)
                except Exception:
                    lbl_mtr = lbl_mts = None

                # 只有在有标签时才做分割 meta（否则跳过）
                if (lbl_mtr is not None) and (vi_mtr.shape[0] > 0) and (vi_mts.shape[0] > 0):
                    t0 = time.time()
                    # 1) 在 mtr 上计算分割损失（教师/内任务），这里对分割 head 做一阶求导
                    with autocast('cuda', enabled=args.use_amp):
                        _, seg_mtr, *_ = model(vi_mtr, ir_mtr, return_lists=True)
                        Lseg_mtr = ce_loss(seg_mtr, lbl_mtr)

                    grads_S = torch.autograd.grad(
                        Lseg_mtr, S_params,
                        create_graph=True, retain_graph=True, allow_unused=True
                    )
                    grads_S = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_S, S_params)]

                    # 构造更新后的 S' 参数字典
                    all_params = make_params_dict(m)
                    updated = dict(all_params)
                    for n, p, g in zip(S_names, S_params, grads_S):
                        updated[n] = p - args.inner_lr * g

                    # 在 mts 上用 S' 前向，计算 Lseg_mts（或其他评价 loss）
                    with autocast('cuda', enabled=args.use_amp):
                        _, seg_mts, *_ = functional_call(
                            model, updated, args=(vi_mts, ir_mts), kwargs={"return_lists": True}
                        )
                        Lseg_mts = ce_loss(seg_mts, lbl_mts)

                    # 计算 Lseg_mts 对原始 S_params 的二阶梯度
                    grads_meta_S = torch.autograd.grad(
                        Lseg_mts, S_params,
                        retain_graph=False, allow_unused=True
                    )
                    grads_meta_S = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_meta_S, S_params)]

                    # 用 opt_S 更新分割 head（只更新 seg head）
                    if opt_S is not None:
                        opt_S.zero_grad(set_to_none=True)
                        for p, g in zip(S_params, grads_meta_S):
                            p.grad = g.detach().clone()
                        opt_S.step()

                    t1 = time.time()
                    inner_time_seg = t1 - t0
                else:
                    inner_time_seg = 0.0
                pbar.set_postfix(inner_time_seg=f"{inner_time_seg:.2f}s", Lf=Lf.item(), Lseg=loss_seg.item(), Lout=L_out.item())

            # ---- 统计 ----
            epoch_loss["Lf"]   += Lf.item()
            # epoch_loss["Lg"]   += Lg.item()
            epoch_loss["Lseg"] += loss_seg.item()
            # epoch_loss["Lout"] += L_out.item()
            count += 1

        # 记录平均 loss
        for k in epoch_loss:
            loss_history[k].append(epoch_loss[k] / max(1, count))

        # 保存模型
        save_path = os.path.join(args.save_dir, f"msrs_seg_b1_meta_epochs100.pth")
        torch.save(model.state_dict(), save_path)

    # -------- 训练完成后绘制 loss 曲线 --------
    plt.figure(figsize=(10,6))
    for k, v in loss_history.items():
        plt.plot(range(1, args.epochs+1), v, label=k)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss Curves"); plt.legend(); plt.grid(True)
    fig_path = os.path.join(args.save_dir, "msrs_seg_b1_meta_epochs100_loss_curve.png")
    plt.savefig(fig_path)
    print(f"[INFO] Loss 曲线已保存到 {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="runs_meta")
    parser.add_argument("--ckpt_path", type=str, default="runs/segformer_b1_backbone_weights.pth")
    parser.add_argument("--num_classes", type=int, default=17)

    parser.add_argument("--lr_f", type=float, default=5e-5)
    parser.add_argument("--lr_mfe", type=float, default=5e-5)
    parser.add_argument("--lr_seg", type=float, default=5e-5)
    parser.add_argument("--lr_all", type=float, default=5e-4, help="整体快速优化学习率")

    parser.add_argument("--lambda_g", type=float, default=0.1)
    parser.add_argument("--inner_lr", type=float, default=1e-5)
    parser.add_argument("--inner_every", type=int, default=3)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--inner_warmup", type=int, default=1, help="前 W 个 epoch 不做内更新")

    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()
    train(args)
