import torch
from torch import nn


class reflect_conv(nn.Module):
    """Reflection padding followed by convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        """Create a reflection-padded convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            pad: Reflection padding size.

        Returns:
            None.
        """
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        """Apply the reflection-padded convolution.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after padding and convolution.
        """
        out = self.conv(x)
        return out


def gradient(input):
    """Compute Sobel image gradients.

    Args:
        input: Image tensor with shape ``[B, 1, H, W]``.

    Returns:
        Absolute Sobel gradient tensor with the same spatial shape.
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ], device=input.device, dtype=input.dtype).reshape(1, 1, 3, 3)
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ], device=input.device, dtype=input.dtype).reshape(1, 1, 3, 3)
    filter1 = filter1.to(device=input.device, dtype=input.dtype)
    filter2 = filter2.to(device=input.device, dtype=input.dtype)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient



def clamp(value, min=0., max=1.0):
    """Clamp tensor values into a valid range.

    Args:
        value: Input tensor.
        min: Minimum allowed value.
        max: Maximum allowed value.

    Returns:
        Clamped tensor.
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """Convert an RGB tensor to YCbCr components.

    Args:
        rgb_image: RGB image tensor with shape ``[3, H, W]``.

    Returns:
        Tuple ``(Y, Cb, Cr)`` with one channel per tensor.
    """
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """Convert YCbCr components back to RGB.

    Args:
        Y: Luminance tensor with shape ``[1, H, W]``.
        Cb: Blue-difference chroma tensor with shape ``[1, H, W]``.
        Cr: Red-difference chroma tensor with shape ``[1, H, W]``.

    Returns:
        RGB image tensor with shape ``[3, H, W]``.
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out
