# codes/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import gradient

class FusionLoss(nn.Module):
    """Fusion loss combining gradient preservation and pixel intensity terms."""

    def __init__(self, grad_weight=50.0, pix_weight=20.0):
        """Create a fusion loss module.

        Args:
            grad_weight: Weight for the Sobel-gradient L1 loss.
            pix_weight: Weight for the pixel-intensity L1 loss.

        Returns:
            None.
        """
        super(FusionLoss, self).__init__()
        self.grad_weight = grad_weight
        self.pix_weight = pix_weight

    def forward(self, fused, vi, ir):
        """Compute fusion loss.

        Args:
            fused: Fused image tensor with shape ``[B, 1, H, W]``.
            vi: Visible luminance tensor with shape ``[B, 1, H, W]``.
            ir: Infrared tensor with shape ``[B, 1, H, W]``.

        Returns:
            Tuple ``(total_loss, gradient_loss, pixel_loss)``.
        """
        # Preserve the strongest modality edge at each pixel.
        loss_grad = F.l1_loss(gradient(fused), torch.max(gradient(vi), gradient(ir)))
        # Preserve the strongest modality intensity at each pixel.
        loss_pix  = F.l1_loss(fused, torch.max(vi, ir))
        
        total_loss = self.grad_weight * loss_grad + self.pix_weight * loss_pix
        return total_loss, loss_grad, loss_pix

def ce_loss(logits, labels):
    """Compute semantic segmentation cross-entropy loss.

    Args:
        logits: Prediction logits with shape ``[B, C, H, W]``.
        labels: Ground-truth labels with shape ``[B, H, W]``.

    Returns:
        Scalar cross-entropy loss tensor.
    """
    return nn.CrossEntropyLoss()(logits, labels)
