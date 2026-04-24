import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from data_pipeline.dataloader import VIFSDataset
from nets.segformer import SegFormer
from utils.common import YCrCb2RGB
from utils.utils_logger import get_logger
from config_loader import load_configs
from utils.metrics import SegmentationMetric  # 新增

def save_image(tensor, path):
    """
    将模型输出的张量转换为图像并保存。
    """
    tensor = tensor.detach().cpu().clamp(0, 1)  # 限制在0-1防止溢出
    tensor = (tensor * 255).byte()
    if tensor.shape[0] == 1:  # 灰度图
        img = Image.fromarray(tensor[0].numpy())
    else:  # RGB彩色图
        img = Image.fromarray(tensor.permute(1, 2, 0).numpy())
    img.save(path)

def test_model():
    cfg, params = load_configs()
    dataset_cfg = cfg['dataset']
    exp = cfg['backbone']['phi']
    p_test = params['test']

    # 获取文件夹分级用的架构名
    model_arch = cfg.get('model_name')
    
    # === 1. 动态构建路径 ===
    exp_name = f"{dataset_cfg['name']}_{exp}"

    # ---------------------------------------------------------
    # 智能权重寻址逻辑
    # ---------------------------------------------------------
    weight_target = cfg['test']['checkpoint_name']
    
    if weight_target.endswith('.pth'):
        # 测试模型带完整后缀，直接使用该文件名
        weight_file = weight_target
    else:
        # 自动拼接：自动组合 数据集_phi_标识.pth
        weight_file = f"{exp_name}_{weight_target}.pth"
    # ---------------------------------------------------------
    
    # 推断模型的完整加载路径，增加 model_name 层级
    train_save_dir = os.path.join(cfg['train']['save_base_dir'], model_arch, exp_name)
    model_path = os.path.join(train_save_dir, weight_file)
    
    # 动态创建测试结果输出目录
    save_dir = os.path.join(cfg['test']['save_base_dir'], model_arch, f"{exp_name}_results")
    seg_save_dir = f"{save_dir}_seg"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)

    # === [新增] 初始化 Logger ===
    log_file_path = os.path.join(save_dir, "test_evaluation.log")
    logger = get_logger(name="Test", log_file=log_file_path)

    logger.info("================ 开启测试评估任务 ================")
    logger.info(f"数据集: {dataset_cfg['name']}")
    logger.info(f"加载模型权重: {model_path}")
    logger.info(f"融合图像保存至: {save_dir}")
    logger.info(f"分割掩码保存至: {seg_save_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 2. 加载数据 (注入测试目录) ===
    base_path = os.path.join(dataset_cfg['root_dir'], dataset_cfg['name'])
    test_paths = {
        'vi_dir': os.path.join(base_path, 'vi', 'test'),
        'ir_dir': os.path.join(base_path, 'ir', 'test'),
        'label_dir': os.path.join(base_path, 'label', 'test'),
    }
    
    test_dataset = VIFSDataset(mode='test', resize_size=tuple(p_test['resize_size']), **test_paths)
    test_loader = DataLoader(test_dataset, batch_size=p_test['batch_size'], shuffle=False, num_workers=p_test['num_workers'])

    # === 3. 初始化并加载模型 ===
    model = SegFormer(num_classes=p_test['num_classes'])
    
    if os.path.exists(model_path):
        # 兼容包含 "module." 前缀的 DataParallel 保存格式
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        logger.info("模型权重加载成功。")
    else:
        logger.warning(f"未找到模型权重文件: {model_path}，当前使用随机权重进行推断！")
        
    model = model.to(device)
    model.eval()

    # === 4. 执行测试与 IoU 统计 ===
    num_classes = p_test['num_classes']
    metric = SegmentationMetric(num_classes)

    with torch.no_grad():
        for idx, (vi_y, ir_image, label_tensor, name, cr, cb) in tqdm(enumerate(test_loader), total=len(test_loader)):
            vi_y = vi_y.to(device)
            ir_image = ir_image.to(device)
            label_tensor = label_tensor.to(device)
            cr = cr.to(device)
            cb = cb.to(device)

            # 前向传播提取特征
            fused_img, seg_img = model(vi_y, ir_image)

            # 1. 重构彩色融合图并保存 (融合通道Y + 原始色彩CrCb)
            fused_rgb = YCrCb2RGB(fused_img[0], cr[0], cb[0])
            save_image(fused_rgb, os.path.join(save_dir, name[0]))

            # 2. 提取分割预测并保存
            seg_pred = torch.argmax(seg_img, dim=1)   # [B,H,W]
            seg_img_np = seg_pred[0].cpu().numpy().astype(np.uint8)
            seg_pil = Image.fromarray(seg_img_np)
            seg_pil.save(os.path.join(seg_save_dir, f'{name[0]}'))

            # 3. 统计逐像素交并比 (Intersection over Union)
            pred_np = seg_pred.cpu().numpy()
            label_np = label_tensor.cpu().numpy()

            metric.add_batch(label_np, pred_np)
    
    # 提取交集和并集，无缝衔接原有的日志输出
    intersection = np.diag(metric.confusion_matrix)
    union = np.sum(metric.confusion_matrix, axis=1) + np.sum(metric.confusion_matrix, axis=0) - intersection
    
    # === 5. 输出指标统计 ===
    logger.info("="*30)
    ious = []
    for cls in range(num_classes):
        if union[cls] > 0:
            iou = intersection[cls] / union[cls]
            ious.append(iou)
            logger.info(f"Class {cls:02d}: IoU = {iou:.4f} ({intersection[cls]}/{union[cls]})")
        else:
            logger.info(f"Class {cls:02d}: no samples in GT")
            ious.append(np.nan)

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious)
    logger.info("="*30)
    logger.info(f"Final Mean IoU (mIoU): {miou:.4f}")
    logger.info("================ 测试流程执行完毕 ================")

if __name__ == '__main__':
    test_model()