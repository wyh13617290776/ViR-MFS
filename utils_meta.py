# utils_meta.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import gradient

# -------- 权重加载 --------
def _strip_module(sd: dict):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _add_module(sd: dict):
    return { (k if k.startswith("module.") else f"module.{k}"): v for k, v in sd.items() }

def load_partial_weights(model, weight_path, device="cuda"):
    if not os.path.exists(weight_path):
        print(f"[WARN] 未找到预训练权重: {weight_path}")
        return model

    # 尽量使用安全加载（老版本PyTorch不支持就回退）
    try:
        ckpt = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unexpected checkpoint format")

    model_dict = model.state_dict()

    # 三种形式分别试一下
    cands = [("raw", sd), ("strip", _strip_module(sd)), ("add", _add_module(sd))]

    best_name, best_matched, best_dict = None, -1, None
    for name, cand in cands:
        matched = {k: v for k, v in cand.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(matched) > best_matched:
            best_matched = len(matched)
            best_name, best_dict = name, matched

    print(f"[INFO] 预训练权重: {weight_path}")
    print(f"[INFO] key 形式: {best_name}, 匹配到的参数: {best_matched}/{len(model_dict)}")

    model_dict.update(best_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

# def load_partial_weights(model, weight_path, device="cuda"):
#     """
#     加载部分预训练权重：
#     - 支持 state_dict 或 {'state_dict': ...} 格式
#     - 只加载名字和 shape 都匹配的参数
#     - 打印匹配数量，允许部分未加载
#     """
#     if not os.path.exists(weight_path):
#         print(f"[WARN] 未找到预训练权重: {weight_path}")
#         return model

#     try:
#         checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
#     except TypeError:
#         checkpoint = torch.load(weight_path, map_location=device)

#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         pretrained_dict = checkpoint["state_dict"]
#     elif isinstance(checkpoint, dict):
#         pretrained_dict = checkpoint
#     else:
#         raise ValueError("Unexpected checkpoint format")

#     model_dict = model.state_dict()
#     matched_dict = {k: v for k, v in pretrained_dict.items()
#                     if k in model_dict and v.size() == model_dict[k].size()}

#     print(f"[INFO] 预训练权重: {weight_path}")
#     print(f"[INFO] 匹配到的参数: {len(matched_dict)}/{len(model_dict)}")

#     model_dict.update(matched_dict)
#     model.load_state_dict(model_dict, strict=False)
#     return model

# -------- 参数相关 --------
# def get_fusion_param_names(m: nn.Module):
#     keep_prefix = ("shallow1", "shallow2", "seg1", "seg2", "seg3", "fusion_task_head")
#     names = [n for n, p in m.named_parameters() if any(n.startswith(k) for k in keep_prefix)]
#     return names
def get_fusion_param_names(m: nn.Module):
    keep_prefix = ("f0", "f1", "f2", "f3", "fusion_head")
    names = [n for n, p in m.named_parameters() if any(n.startswith(k) for k in keep_prefix)]
    return names

def make_params_dict(m: nn.Module):
    return {name: p for name, p in m.named_parameters()}

def merge_updated_params(params_all, names_to_update, updates, inner_lr):
    new_params = dict(params_all)
    for name, g in zip(names_to_update, updates):
        new_params[name] = params_all[name] - inner_lr * g
    return new_params

# -------- 损失函数 --------
def fusion_loss(fused, vi, ir):
    loss_grad = F.l1_loss(gradient(fused), torch.max(gradient(vi), gradient(ir)))
    loss_pix  = F.l1_loss(fused, torch.max(vi, ir))
    return 50.0 * loss_grad + 20.0 * loss_pix, loss_grad, loss_pix

def ce_loss(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)

# -------- 其他工具 --------
def split_mtr_mts(x: torch.Tensor):
    b = x.shape[0]
    mid = b // 2
    if mid == 0:
        return x, x
    return x[:mid], x[mid:]
