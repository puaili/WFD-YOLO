# Ultralytics YOLOv12, AGPL-3.0 license
"""WFD-YOLO modules described in the accompanying surgical-instrument paper."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("WFCA", "SFPA", "DGFA", "WFCAParallel", "ParallelFusion")


class _AdaptiveBranchMixer(nn.Module):
    """
    Learn a convex mixture of the untouched YOLO path and an innovation path.

    The scalar is deliberately initialized in favour of the baseline path.
    Unlike an unconstrained residual scale, the sigmoid parameterization keeps
    both branch weights non-negative and summing to one. If an innovation is
    unhelpful, optimization can drive its weight towards zero and recover the
    original YOLOv12n computation.
    """

    def __init__(self, innovation_init: float = 0.05):
        super().__init__()
        if not 0.0 < innovation_init < 1.0:
            raise ValueError(f"innovation_init must be in (0, 1), received {innovation_init}")
        logit = math.log(innovation_init / (1.0 - innovation_init))
        self.innovation_logit = nn.Parameter(torch.tensor(float(logit)))

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        innovation = self.innovation_logit.sigmoid()
        return 1.0 - innovation, innovation

    def forward(self, baseline: torch.Tensor, innovation: torch.Tensor) -> torch.Tensor:
        _, innovation_weight = self.weights()
        return torch.lerp(baseline, innovation, innovation_weight.to(dtype=baseline.dtype))


class _ZeroInitResidualAdapter(nn.Module):
    """
    Make an innovation branch exactly identical to its baseline at step zero.

    A zero-initialized depthwise 1x1 adapter learns a per-channel residual gain.
    Its weights receive gradients immediately, while the downstream pretrained
    YOLO layers initially see the exact baseline tensor.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, baseline: torch.Tensor, innovation: torch.Tensor) -> torch.Tensor:
        return baseline + self.projection(innovation - baseline)

    def mean_absolute_gain(self) -> float:
        return float(self.projection.weight.detach().abs().mean())


class _ChannelMLP(nn.Module):
    """Shared squeeze MLP used to derive channel weights."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fft2_float(x: torch.Tensor) -> torch.Tensor:
    """Run FFT in FP32 because CUDA half FFT only supports power-of-two sizes."""
    return torch.fft.fft2(x.float(), norm="ortho")


def _ifft2_real(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return the real component of an orthonormal inverse FFT."""
    return torch.fft.ifft2(x, norm="ortho").real.to(dtype=dtype)


def _haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Differentiable 2-D Haar DWT implemented by polyphase slicing."""
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _haar_idwt2(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`_haar_dwt2`, cropped to the original spatial size."""
    x00 = (ll - lh - hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll + lh + hl + hh) * 0.5
    out = torch.empty(
        *ll.shape[:-2],
        ll.shape[-2] * 2,
        ll.shape[-1] * 2,
        device=ll.device,
        dtype=ll.dtype,
    )
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out[..., : output_size[0], : output_size[1]]


class WFCA(nn.Module):
    """
    Wavelet-Fourier Cooperative Attention.

    The output follows the paper's residual gated formulation:
    ``O = X + g * X_w + (1 - g) * X_f``.
    """

    def __init__(self, channels: int, reduction: int = 16, wavelet_ratio: float = 0.25):
        super().__init__()
        hidden = max(int(channels * wavelet_ratio), 8)
        self.wavelet_fusion = nn.Sequential(
            Conv(channels * 4, hidden, 1),
            DWConv(hidden, hidden, 3),
            nn.Conv2d(hidden, channels * 4, 1, bias=False),
        )
        self.frequency_mlp = _ChannelMLP(channels, reduction)
        gate_hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, gate_hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def _wavelet_branch(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = _haar_dwt2(x)
        bands = self.wavelet_fusion(torch.cat((ll, lh, hl, hh), dim=1))
        ll, lh, hl, hh = bands.chunk(4, dim=1)
        return _haar_idwt2(ll, lh, hl, hh, x.shape[-2:])

    def _frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        spectrum = _fft2_float(x)
        magnitude = spectrum.abs()
        avg = magnitude.mean(dim=(-2, -1), keepdim=True).to(dtype)
        maximum = magnitude.amax(dim=(-2, -1), keepdim=True).to(dtype)
        attention = torch.sigmoid(self.frequency_mlp(avg) + self.frequency_mlp(maximum)).float()
        return _ifft2_real(spectrum * attention, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w = self._wavelet_branch(x)
        x_f = self._frequency_branch(x)
        gate = self.gate(torch.cat((x_w, x_f), dim=1))
        return x + gate * x_w + (1.0 - gate) * x_f


class WFCAParallel(nn.Module):
    """
    Identity-preserving parallel WFCA.

    The baseline feature is never replaced. A learnable convex gate chooses
    between the untouched YOLOv12n feature and the WFCA-enhanced feature.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        wavelet_ratio: float = 0.25,
        innovation_init: float = 0.05,
    ):
        super().__init__()
        self.innovation = WFCA(channels, reduction=reduction, wavelet_ratio=wavelet_ratio)
        self.adapter = _ZeroInitResidualAdapter(channels)
        self.mixer = _AdaptiveBranchMixer(innovation_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        innovation = self.adapter(x, self.innovation(x))
        return self.mixer(x, innovation)

    def branch_weights(self) -> tuple[float, float]:
        baseline, innovation = self.mixer.weights()
        return float(baseline.detach()), float(innovation.detach())

    def adapter_gain(self) -> float:
        return self.adapter.mean_absolute_gain()


class SFPA(nn.Module):
    """Subband-guided Fusion Purification Attention."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        sigma: float = 1.0,
        threshold: float = 0.5,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.frequency_mlp = _ChannelMLP(channels, reduction)
        self.spatial_attention = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.threshold = float(threshold)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        kernel = self._gaussian_kernel(size=5, sigma=sigma)
        self.register_buffer("gaussian_kernel", kernel[None, None], persistent=False)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        return torch.outer(kernel_1d, kernel_1d)

    def _gaussian_filter(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.gaussian_kernel.to(device=x.device, dtype=x.dtype).expand(channels, 1, -1, -1)
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        spectrum = _fft2_float(x)
        real, imag = spectrum.real, spectrum.imag
        avg = real.mean(dim=(-2, -1), keepdim=True).to(dtype)
        maximum = real.amax(dim=(-2, -1), keepdim=True).to(dtype)
        channel_weight = torch.sigmoid(self.frequency_mlp(avg) + self.frequency_mlp(maximum)).float()
        reconstructed = _ifft2_real(torch.complex(real * channel_weight, imag * channel_weight), dtype)

        spatial_weight = self.spatial_attention(
            torch.cat(
                (
                    reconstructed.mean(dim=1, keepdim=True),
                    reconstructed.amax(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        purified = x * spatial_weight
        smoothed = self._gaussian_filter(purified)
        gated = torch.where(purified.abs() < self.threshold, smoothed, purified)
        return purified + self.residual_scale * gated


class _DepthwiseSeparableConv(nn.Module):
    """Lightweight 3x3 channel mapper used by DGFA."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.block = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DGFA(nn.Module):
    """
    Dual-domain Gated Fusion Attention.

    Input order is ``[spatial_lateral, semantic_deep]``. The output channel
    count defaults to the sum of both inputs so it can replace ``Concat``.
    """

    def __init__(
        self,
        channels: Sequence[int],
        out_channels: int | None = None,
        align_ratio: float = 0.375,
        reduction: int = 16,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"DGFA expects two inputs, received channels={channels}")
        spatial_channels, semantic_channels = (int(c) for c in channels)
        out_channels = int(out_channels or (spatial_channels + semantic_channels))
        aligned = max(8, int(min(spatial_channels, semantic_channels) * align_ratio))
        aligned = int(math.ceil(aligned / 8) * 8)

        self.spatial_map = Conv(spatial_channels, aligned, 1)
        self.semantic_map = nn.Sequential(
            _DepthwiseSeparableConv(semantic_channels, aligned),
            WFCA(aligned, reduction=reduction),
        )
        self.gate = nn.Sequential(nn.Conv2d(aligned * 2, aligned, 1, bias=True), nn.Sigmoid())
        self.output_map = Conv(aligned, out_channels, 1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        spatial, semantic = inputs
        spatial = self.spatial_map(spatial)
        semantic = self.semantic_map(semantic)
        guide = torch.cat((spatial, semantic), dim=1)
        gate = self.gate(guide)
        fused = gate * spatial + (1.0 - gate) * semantic
        return self.output_map(fused)


class ParallelFusion(nn.Module):
    """
    Preserve the original Concat while learning an SFPA+DGFA alternative.

    Input ordering follows the YOLOv12 YAML: the second tensor is the lateral
    skip feature and the first tensor is the resized feature from the main
    path. The baseline branch is the exact original ``torch.cat`` operation.
    """

    def __init__(
        self,
        channels: Sequence[int],
        dimension: int = 1,
        align_ratio: float = 0.375,
        reduction: int = 16,
        innovation_init: float = 0.05,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"ParallelFusion expects two inputs, received channels={channels}")
        self.dimension = int(dimension)
        main_channels, lateral_channels = (int(c) for c in channels)
        self.purifier = SFPA(lateral_channels, reduction=reduction)
        self.innovation = DGFA(
            [lateral_channels, main_channels],
            out_channels=main_channels + lateral_channels,
            align_ratio=align_ratio,
            reduction=reduction,
        )
        self.adapter = _ZeroInitResidualAdapter(main_channels + lateral_channels)
        self.mixer = _AdaptiveBranchMixer(innovation_init)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"ParallelFusion expects two tensors, received {len(inputs)}")
        main, lateral = inputs
        baseline = torch.cat((main, lateral), dim=self.dimension)
        innovation = self.innovation((self.purifier(lateral), main))
        innovation = self.adapter(baseline, innovation)
        return self.mixer(baseline, innovation)

    def branch_weights(self) -> tuple[float, float]:
        baseline, innovation = self.mixer.weights()
        return float(baseline.detach()), float(innovation.detach())

    def adapter_gain(self) -> float:
        return self.adapter.mean_absolute_gain()
