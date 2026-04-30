# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .wtconv2d import WTConv2d_VIF

class MLP(nn.Module):
    """Linear projection used by the SegFormer decoder."""

    def __init__(self, input_dim=2048, embed_dim=768):
        """Create a feature projection layer.

        Args:
            input_dim: Number of input feature channels.
            embed_dim: Number of output embedding channels.

        Returns:
            None.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """Project a 2D feature map into token embeddings.

        Args:
            x: Feature tensor with shape ``[B, C, H, W]``.

        Returns:
            Tensor with shape ``[B, H*W, embed_dim]``.
        """
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    """Convolution, batch normalization, and activation block."""

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        """Create a convolution block.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            k: Kernel size.
            s: Stride.
            p: Padding.
            g: Number of convolution groups.
            act: Activation flag or activation module.

        Returns:
            None.
        """
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """Apply convolution, normalization, and activation.

        Args:
            x: Input feature tensor.

        Returns:
            Processed feature tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x: Input feature tensor.

        Returns:
            Processed feature tensor.
        """
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """SegFormer decoder head for dense prediction."""

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        """Create a multi-scale SegFormer head.

        Args:
            num_classes: Number of output channels/classes.
            in_channels: Channel dimensions of the four backbone stages.
            embedding_dim: Shared embedding dimension for decoder fusion.
            dropout_ratio: Dropout probability before prediction.

        Returns:
            None.
        """
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        """Decode multi-scale backbone features.

        Args:
            inputs: List of four feature tensors from low to high semantic level.

        Returns:
            Dense logits tensor at the first feature scale.
        """
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class SegFormer(nn.Module):
    """Visible-infrared fusion and semantic segmentation network."""

    def __init__(
        self,
        num_classes: int,
        pretrained=False,
        backbone_phi="b0",
        pretrained_dir="model_data",
        wavelet_config=None,
    ):
        """Create the ViR-MFS SegFormer model.

        Args:
            num_classes: Number of semantic segmentation classes.
            pretrained: Whether to load a pretrained SegFormer backbone.
            backbone_phi: SegFormer backbone variant.
            pretrained_dir: Directory containing pretrained backbone weights.
            wavelet_config: Optional MWFM configuration dictionary. Supported
                keys include ``kernel_size``, ``wt_levels``, ``wt_type``, and
                high-frequency injection controls.

        Returns:
            None.
        """
        super(SegFormer, self).__init__()

        backbone_weight_path = None
        if pretrained:
            backbone_weight_path = f"{pretrained_dir}/segformer_{backbone_phi}_backbone_weights.pth"
        
        ext_backbones = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5
        }
        
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[backbone_phi]
        self.backbone = ext_backbones[backbone_phi](pretrained_path=backbone_weight_path)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[backbone_phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.fusion_head = SegFormerHead(2, self.in_channels, self.embedding_dim)

        wavelet_config = wavelet_config or {}
        self.f0 = WTConv2d_VIF(in_channels=self.in_channels[0], out_channels=self.in_channels[0], **wavelet_config)
        self.f1 = WTConv2d_VIF(in_channels=self.in_channels[1], out_channels=self.in_channels[1], **wavelet_config)
        self.f2 = WTConv2d_VIF(in_channels=self.in_channels[2], out_channels=self.in_channels[2], **wavelet_config)
        self.f3 = nn.Conv2d(in_channels=self.in_channels[3]*2, out_channels=self.in_channels[3], kernel_size=1, stride=1, padding=0)

        # ---------------- wo_MWFM ----------------
        # Convolutional fallback used by ablation experiments.
        # self.f0 = nn.Conv2d(in_channels=self.in_channels[0] * 2, out_channels=self.in_channels[0], kernel_size=3, padding=1)
        # self.f1 = nn.Conv2d(in_channels=self.in_channels[1] * 2, out_channels=self.in_channels[1], kernel_size=3, padding=1)
        # self.f2 = nn.Conv2d(in_channels=self.in_channels[2] * 2, out_channels=self.in_channels[2], kernel_size=3, padding=1)
        # self.f3 = nn.Conv2d(in_channels=self.in_channels[3] * 2, out_channels=self.in_channels[3], kernel_size=1)
        # ---------------- wo_MWFM ----------------
    def forward(self, inputs, inputs_ir,return_lists=False):
        """Run fusion and segmentation forward pass.

        Args:
            inputs: Visible luminance tensor with shape ``[B, 1, H, W]``.
            inputs_ir: Infrared tensor with shape ``[B, 1, H, W]``.
            return_lists: Whether to return compatibility placeholders used by
                older meta-learning code.

        Returns:
            ``(fused_image, segmentation_logits)`` by default. If
            ``return_lists`` is true, returns
            ``(fused_image, segmentation_logits, fused_image, segmentation_logits)``.
        """
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(torch.cat([inputs]*3, dim=1))
        x_ir = self.backbone.forward(torch.cat([inputs_ir]*3, dim=1))
        # Fuse multi-scale visible and infrared features with MWFM on shallow stages.
        # ---------------- modify_1 ----------------
        f_feature = list(x)
        f_feature[0] = self.f0(x[0], x_ir[0])
        f_feature[1] = self.f1(x[1], x_ir[1])
        f_feature[2] = self.f2(x[2], x_ir[2])
        f_feature[3] = self.f3(torch.cat([x[3],x_ir[3]],dim=1))
        # ---------------- modify_1 ----------------
        
        # Symmetric MWFM ablation path.
        # ---------------- modify_2 ----------------
        # f_feature = x
        # f_feature[0] = self.f0(x[0], x_ir[0]) + self.f0(x_ir[0], x[0])
        # f_feature[1] = self.f1(x[1], x_ir[1])+ self.f1(x_ir[1], x[1])
        # f_feature[2] = self.f2(x[2], x_ir[2])+ self.f2(x_ir[2], x[2])
        # f_feature[3] = self.f3(torch.cat([x[3],x_ir[3]],dim=1))
        # ---------------- modify_2 ----------------

        # Symmetric MWFM ablation path.
        # ---------------- wo_MWFM ----------------
        # Concatenate both modality features for convolutional ablation.
        # f_feature = [
        #     self.f0(torch.cat([x[0], x_ir[0]], dim=1)),
        #     self.f1(torch.cat([x[1], x_ir[1]], dim=1)),
        #     self.f2(torch.cat([x[2], x_ir[2]], dim=1)),
        #     self.f3(torch.cat([x[3], x_ir[3]], dim=1))
        # ]
        # ---------------- wo_MWFM ----------------

        seg = self.decode_head.forward(f_feature)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)
        fus_map = self.fusion_head.forward(f_feature)
        fus_map = F.interpolate(fus_map, size=(H, W), mode='bilinear', align_corners=True)
        fus_img = fus_map[:,0:1,:,:]*inputs + fus_map[:,1:,:,:]*(inputs_ir)
        if return_lists:
            return fus_img, seg, fus_img, seg
        return fus_img, seg



#------------------------------------------------------------------------------#
# Single-modal ablation model.


class SegFormer_s_modal(nn.Module):
    """Single-modal SegFormer segmentation model used for ablations."""

    def __init__(self, num_classes=21, pretrained=False, backbone_phi="b0", pretrained_dir="model_data"):
        """Create a single-modal SegFormer.

        Args:
            num_classes: Number of semantic segmentation classes.
            pretrained: Whether to load pretrained backbone weights.
            backbone_phi: SegFormer backbone variant.
            pretrained_dir: Directory containing pretrained backbone weights.

        Returns:
            None.
        """
        super(SegFormer_s_modal, self).__init__()

        backbone_weight_path = None
        if pretrained:
            backbone_weight_path = f"{pretrained_dir}/segformer_{backbone_phi}_backbone_weights.pth"
        
        ext_backbones = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5
        }

        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[backbone_phi]
        self.backbone = ext_backbones[backbone_phi](pretrained_path=backbone_weight_path)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[backbone_phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)


    def forward(self, inputs):
        """Run single-modal semantic segmentation.

        Args:
            inputs: Input luminance tensor with shape ``[B, 1, H, W]``.

        Returns:
            Segmentation logits with shape ``[B, num_classes, H, W]``.
        """
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(torch.cat([inputs] * 3, dim=1))

        seg = self.decode_head.forward(x)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)
        return seg
