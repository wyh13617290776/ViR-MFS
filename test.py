# test_meta.py
import os
import torch
from torch.utils.data import DataLoader
from dataloder import vifs_dataloder_test
from models.final_model import Fusion_Seg_Model
import numpy as np
import tqdm
from PIL import Image
from models.common import YCrCb2RGB

# ---------------------------
# 工具：保存图像
# ---------------------------
def save_image(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255).byte()
    if tensor.shape[0] == 1:  # [1,H,W] 灰度
        img = Image.fromarray(tensor[0].numpy())
    else:                      # [3,H,W] RGB
        img = Image.fromarray(tensor.permute(1, 2, 0).numpy())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# ---------------------------
# 工具：自适配加载权重（自动处理 module. 前缀）
# ---------------------------
def _strip_module(sd: dict):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _add_module(sd: dict):
    return { (k if k.startswith("module.") else f"module.{k}"): v for k, v in sd.items() }

def load_weights_adaptive(model: torch.nn.Module, weight_path: str, device="cuda"):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}")

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
    cands = [("raw", sd), ("strip", _strip_module(sd)), ("add", _add_module(sd))]

    best_name, best_cnt, best_dict = None, -1, None
    for name, cand in cands:
        matched = {k: v for k, v in cand.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(matched) > best_cnt:
            best_cnt, best_name, best_dict = len(matched), name, matched

    print(f"[INFO] 加载权重: {weight_path}")
    print(f"[INFO] key 形式: {best_name}, 匹配到的参数: {best_cnt}/{len(model_dict)}")
    model_dict.update(best_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

# ---------------------------
# 主测试函数
# ---------------------------
def test_model(model_path='runs_meta/finalv1seg_meta_epoch0.pth',
               batch_size=1,
               save_dir='test_results_meta',
               num_classes=21,
               use_dataparallel=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    test_dataset = vifs_dataloder_test()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 结果目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '_seg', exist_ok=True)

    # 模型
    base_model = Fusion_Seg_Model()  # 与训练时一致（训练保存的是 model.state_dict()）
    if use_dataparallel:
        model = torch.nn.DataParallel(base_model).to(device)
        # DataParallel 包裹后 state_dict 带 "module."，因此适配加载
        model = load_weights_adaptive(model, model_path, device=device)
    else:
        model = base_model.to(device)
        model = load_weights_adaptive(model, model_path, device=device)

    model.eval()

    # IoU 累积
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for idx, (vi_image, ir_image, label_tensor, name, cr, cb) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            vi_image = vi_image.to(device)
            ir_image = ir_image.to(device)
            label_tensor = label_tensor.to(device)
            cr = cr.to(device)
            cb = cb.to(device)

            # 前向（与你现有 test.py 一致：返回 fused, seg）
            fused_img, seg_img = model(vi_image, ir_image)

            # 保存融合图（转回 RGB）
            fused_rgb = YCrCb2RGB(fused_img[0], cr[0], cb[0])  # [3,H,W]
            save_image(fused_rgb, os.path.join(save_dir, name[0]))

            # 分割预测与保存
            seg_pred = torch.argmax(seg_img, dim=1)   # [B,H,W]
            seg_np = seg_pred[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(seg_np).save(os.path.join(save_dir + '_seg', f'{name[0]}'))

            # IoU 统计（逐类）
            pred_np = seg_pred.cpu().numpy()
            label_np = label_tensor.cpu().numpy()
            for cls in range(num_classes):
                pred_mask = (pred_np == cls)
                label_mask = (label_np == cls)
                inter = np.logical_and(pred_mask, label_mask).sum()
                uni = np.logical_or(pred_mask, label_mask).sum()
                intersection[cls] += inter
                union[cls] += uni

    # 打印 per-class IoU 和 mIoU
    ious = []
    for cls in range(num_classes):
        if union[cls] > 0:
            iou = intersection[cls] / union[cls]
            ious.append(iou)
            print(f"Class {cls}: IoU = {iou:.4f} ({intersection[cls]}/{union[cls]})")
        else:
            print(f"Class {cls}: no samples in GT")
            ious.append(np.nan)

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else float('nan')
    print(f"\nMean IoU (mIoU): {miou:.4f}")
    print(f'Fused images and segmentation results saved to {save_dir} / {save_dir}_seg')


if __name__ == '__main__':
    # 示例：测试某个 epoch 的权重
    # 你可以直接改 default 的 model_path，或者从命令行自己写个小 wrapper 传参
    test_model(
        model_path='runs/finalv1seg_m3fd.pth',  # 改成你想测试的 epoch
        batch_size=1,
        save_dir='test_results_finalv1seg_m3fd',
        num_classes=17,
        use_dataparallel=True  # 若你的显卡是单卡，也可以=False
    )
