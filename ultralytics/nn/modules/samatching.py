"""Paper-driven Partial ViT building blocks for the SA-Matching DETR reproduction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("PartialPatchEmbed", "PartialViTBlock", "ChannelLayerNorm2d")


class PartialPatchEmbed(nn.Module):
    """DeiT-compatible non-overlapping patch embedding."""

    def __init__(self, c1: int, c2: int = 192, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm over channels at every spatial token."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class PartialViTBlock(nn.Module):
    """DeiT-style block with MHSA applied only to a channel partition.

    All spatial tokens are retained.  For an input ``B x C x H x W``, the
    first ``ceil(r*C)`` channels enter multi-head self-attention while the
    remaining channels use the identity route.  The two partitions are then
    concatenated and followed by a full-channel feed-forward network, matching
    the mechanism described in Eq. (1)-(2) and Fig. 3 of the reference paper.
    """

    def __init__(
        self,
        channels: int,
        partial_ratio: float = 0.5,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not 0.0 < partial_ratio <= 1.0:
            raise ValueError(f"partial_ratio must be in (0, 1], received {partial_ratio}")
        active = int(math.ceil(channels * partial_ratio / num_heads) * num_heads)
        active = min(active, channels)
        if active % num_heads:
            raise ValueError(f"Active channels {active} must be divisible by {num_heads} heads")
        self.channels = int(channels)
        self.active_channels = int(active)
        self.num_heads = int(num_heads)
        self.head_dim = self.active_channels // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.qkv = nn.Linear(self.active_channels, self.active_channels * 3)
        self.attn_drop = float(dropout)
        self.proj = nn.Linear(self.active_channels, self.active_channels)
        self.proj_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def _partial_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        active = tokens[..., : self.active_channels]
        identity = tokens[..., self.active_channels :]
        batch, count, _ = active.shape
        qkv = self.qkv(active).reshape(batch, count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, count, self.active_channels)
        attended = self.proj_drop(self.proj(attended))
        return torch.cat((attended, identity), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {channels}")
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self._partial_attention(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
