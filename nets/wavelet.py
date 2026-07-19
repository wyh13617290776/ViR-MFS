import pywt
import torch
import torch.nn.functional as F


def create_1d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """Create 1D wavelet decomposition and reconstruction filters.

    Args:
        wave: PyWavelets wavelet name.
        in_size: Number of input channels.
        out_size: Number of output channels.
        type: Torch dtype for created filters.

    Returns:
        Tuple ``(dec_filters, rec_filters)``.
    """
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """Create 2D wavelet decomposition and reconstruction filters.

    Args:
        wave: PyWavelets wavelet name.
        in_size: Number of input channels.
        out_size: Number of output channels.
        type: Torch dtype for created filters.

    Returns:
        Tuple ``(dec_filters, rec_filters)`` for grouped convolution.
    """
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_1d_transform(x, filters):
    """Apply grouped 1D discrete wavelet transform.

    Args:
        x: Input tensor with shape ``[B, C, L]``.
        filters: Decomposition filters.

    Returns:
        Wavelet tensor with shape ``[B, C, 2, L/2]``.
    """
    b, c, l = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, l // 2)
    return x


def inverse_1d_wavelet_transform(x, filters):
    """Apply grouped inverse 1D wavelet transform.

    Args:
        x: Wavelet tensor with shape ``[B, C, 2, L/2]``.
        filters: Reconstruction filters.

    Returns:
        Reconstructed tensor with shape ``[B, C, L]``.
    """
    b, c, _, l_half = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_2d_transform(x, filters):
    """Apply grouped 2D discrete wavelet transform.

    Args:
        x: Input tensor with shape ``[B, C, H, W]``.
        filters: Decomposition filters.

    Returns:
        Wavelet tensor with shape ``[B, C, 4, H/2, W/2]``.
    """
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_2d_wavelet_transform(x, filters):
    """Apply grouped inverse 2D wavelet transform.

    Args:
        x: Wavelet tensor with shape ``[B, C, 4, H/2, W/2]``.
        filters: Reconstruction filters.

    Returns:
        Reconstructed tensor with shape ``[B, C, H, W]``.
    """
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
