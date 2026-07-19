import torch
import torch.nn as nn
import torch.nn.functional as F

from . import wavelet

class SpatialGatedFusionBlock(nn.Module):
    """Spatially gated low-frequency fusion block."""

    def __init__(self, channels):
        """Create a gated fusion block.

        Args:
            channels: Number of channels in each modality feature map.

        Returns:
            None.
        """
        super(SpatialGatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            # Output 2*C channels for visible and infrared gates.
            nn.Conv2d(channels // reduction, channels*2, 1, bias=False),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, v, i):
        """Fuse visible and infrared low-frequency features.

        Args:
            v: Visible feature tensor with shape ``[B, C, H, W]``.
            i: Infrared feature tensor with shape ``[B, C, H, W]``.

        Returns:
            Fused feature tensor with shape ``[B, C, H, W]``.
        """
        combined = torch.cat([v, i], dim=1)
        gates = self.attention(combined) # (B, 2*C, H, W)
        gate_v, gate_i = torch.split(gates,self.channels, dim=1) # (B, C, H, W) each
        gated_fusion = gate_v * v + gate_i * i
        gated_fusion = self.final_conv(gated_fusion)
        return gated_fusion


class HighFrequencyInjectionController(nn.Module):
    """Learnable controller for visible/infrared high-frequency injection."""

    def __init__(
        self,
        channels,
        init_source="visible",
        learnable=True,
        init_strength=4.0,
        temperature=1.0,
    ):
        """Create a high-frequency injection controller.

        Args:
            channels: Number of feature channels in each modality.
            init_source: Initial preference for high-frequency bands. Supported
                values are ``visible``, ``infrared``, ``mean``, and ``sum``.
                ``sum`` is initialized as equal visible/infrared weighting in
                the learnable controller to keep the output numerically stable.
            learnable: Whether injection logits are optimized during training.
            init_strength: Absolute logit value used to initialize a strong
                visible or infrared preference.
            temperature: Softmax temperature. Lower values make the modality
                choice sharper.

        Returns:
            None.
        """
        super().__init__()
        if init_source not in {"visible", "infrared", "mean", "sum"}:
            raise ValueError("init_source must be one of: visible, infrared, mean, sum")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.channels = channels
        self.init_source = init_source
        self.temperature = float(temperature)
        logits = torch.zeros(1, 2, channels, 3, 1, 1)

        if init_source == "visible":
            logits[:, 0].fill_(float(init_strength))
            logits[:, 1].fill_(-float(init_strength))
        elif init_source == "infrared":
            logits[:, 0].fill_(-float(init_strength))
            logits[:, 1].fill_(float(init_strength))

        self.logits = nn.Parameter(logits, requires_grad=learnable)

    def forward(self, visible_h, infrared_h):
        """Fuse high-frequency wavelet bands with learnable modality weights.

        Args:
            visible_h: Visible high-frequency tensor with shape
                ``[B, C, 3, H, W]``.
            infrared_h: Infrared high-frequency tensor with shape
                ``[B, C, 3, H, W]``.

        Returns:
            Controlled high-frequency tensor with shape ``[B, C, 3, H, W]``.
        """
        weights = torch.softmax(self.logits / self.temperature, dim=1)
        visible_weight = weights[:, 0]
        infrared_weight = weights[:, 1]
        return visible_weight * visible_h + infrared_weight * infrared_h


class WTConv2d_VIF(nn.Module):
    """Wavelet convolution module for asymmetric visible-infrared fusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type='db1',
        high_frequency_source='visible',
        high_frequency_injection='learnable',
        high_frequency_init_strength=4.0,
        high_frequency_temperature=1.0,
    ):
        """Create the wavelet fusion module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. It must equal
                ``in_channels`` to preserve the original network structure.
            kernel_size: Depth-wise convolution kernel size in wavelet space.
            stride: Optional output stride.
            bias: Whether the base depth-wise convolution uses bias.
            wt_levels: Number of wavelet decomposition levels.
            wt_type: PyWavelets wavelet name.
            high_frequency_source: Source for reconstruction high-frequency
                bands. ``visible`` keeps the original asymmetric design;
                ``infrared``, ``mean``, and ``sum`` are ablation options.
            high_frequency_injection: Injection strategy. ``static`` uses a
                fixed source rule; ``learnable`` uses
                ``HighFrequencyInjectionController`` and learns modality
                weights during training.
            high_frequency_init_strength: Initial logit strength for the
                learnable injection controller.
            high_frequency_temperature: Softmax temperature for the learnable
                injection controller.

        Returns:
            None.
        """
        super(WTConv2d_VIF, self).__init__()

        assert in_channels == out_channels
        if high_frequency_source not in {"visible", "infrared", "mean", "sum"}:
            raise ValueError(
                "high_frequency_source must be one of: visible, infrared, mean, sum"
            )
        if high_frequency_injection not in {"static", "learnable"}:
            raise ValueError("high_frequency_injection must be either static or learnable")

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.high_frequency_source = high_frequency_source
        self.high_frequency_injection = high_frequency_injection

        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels,
                                                                           torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None


        self.fusion = SpatialGatedFusionBlock(self.in_channels)
        self.high_frequency_controller = HighFrequencyInjectionController(
            channels=self.in_channels,
            init_source=high_frequency_source,
            learnable=(high_frequency_injection == "learnable"),
            init_strength=high_frequency_init_strength,
            temperature=high_frequency_temperature,
        )

    def _select_high_frequency(self, visible_h, infrared_h):
        """Select high-frequency bands for inverse wavelet reconstruction.

        Args:
            visible_h: Visible high-frequency tensor with shape
                ``[B, C, 3, H, W]``.
            infrared_h: Infrared high-frequency tensor with shape
                ``[B, C, 3, H, W]``.

        Returns:
            High-frequency tensor with shape ``[B, C, 3, H, W]``.
        """
        if self.high_frequency_injection == "learnable":
            return self.high_frequency_controller(visible_h, infrared_h)
        if self.high_frequency_source == "visible":
            return visible_h
        if self.high_frequency_source == "infrared":
            return infrared_h
        if self.high_frequency_source == "mean":
            return 0.5 * (visible_h + infrared_h)
        return visible_h + infrared_h

    def forward(self, x, y):
        """Run asymmetric wavelet fusion for one feature scale.

        Args:
            x: Visible feature tensor with shape ``[B, C, H, W]``.
            y: Infrared feature tensor with shape ``[B, C, H, W]``.

        Returns:
            Fused feature tensor with shape ``[B, C, H, W]``.
        """

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        y_ll_in_levels = []
        y_h_in_levels = []

        curr_x_ll = x
        curr_y_ll = y

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
                curr_y_ll = F.pad(curr_y_ll, curr_pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            # Both modalities must use the decomposition filter before inverse reconstruction.
            curr_y = wavelet.wavelet_2d_transform(curr_y_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            curr_y_ll = curr_y[:, :, 0, :, :]

            shape_x = curr_x.shape
            shape_y = curr_y.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_y_tag = curr_y.reshape(shape_y[0], shape_y[1] * 4, shape_y[3], shape_y[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_y_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_y_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            curr_y_tag = curr_y_tag.reshape(shape_y)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
            y_ll_in_levels.append(curr_y_tag[:, :, 0, :, :])
            y_h_in_levels.append(curr_y_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_y_ll = y_ll_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = self.fusion(curr_x_ll, curr_y_ll)

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x_h = self._select_high_frequency(curr_x_h, y_h_in_levels.pop())
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    """Learnable scalar calibration module."""

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        """Create a learnable scale tensor.

        Args:
            dims: Parameter tensor dimensions.
            init_scale: Initial multiplicative scale.
            init_bias: Reserved for compatibility; currently unused.

        Returns:
            None.
        """
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        """Apply element-wise scale to an input tensor.

        Args:
            x: Input tensor.

        Returns:
            Scaled tensor with the same shape as ``x``.
        """
        return torch.mul(self.weight, x)
