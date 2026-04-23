# codes/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import gradient

class FusionLoss(nn.Module):
    """
    图像融合损失函数：结合梯度损失(L1)和像素损失(L1)
    """
    def __init__(self, grad_weight=50.0, pix_weight=20.0):
        super(FusionLoss, self).__init__()
        self.grad_weight = grad_weight
        self.pix_weight = pix_weight

    def forward(self, fused, vi, ir):
        # 梯度损失：保持融合图像与最大梯度（可见光/红外）的一致性
        loss_grad = F.l1_loss(gradient(fused), torch.max(gradient(vi), gradient(ir)))
        # 像素损失：保持融合图像与最大像素强度的一致性
        loss_pix  = F.l1_loss(fused, torch.max(vi, ir))
        
        total_loss = self.grad_weight * loss_grad + self.pix_weight * loss_pix
        return total_loss, loss_grad, loss_pix

def ce_loss(logits, labels):
    """
    标准交叉熵损失，用于语义分割任务
    """
    return nn.CrossEntropyLoss()(logits, labels)