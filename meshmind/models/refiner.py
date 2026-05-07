"""MeshMindRefiner — our own from-scratch transformer architecture.

This module is the *original* part of MeshMind: a small transformer that
post-processes latent codes produced by upstream backbones.  It can be applied
to image latents (the output of a Stable Diffusion VAE) or to mesh-latent
tokens before decoding, optionally conditioned on a text embedding via
cross-attention.

The architecture is intentionally compact (a few hundred-thousand to a few
million parameters depending on configuration) so it can be trained on a
single GPU.  Weights are initialised so the refiner behaves as a residual
identity until trained — this lets the rest of the pipeline work out of the
box even without a refiner checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward block (Shazeer, 2020)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class _SelfAttention(nn.Module):
    """Multi-head self attention with RMSNorm-style stability."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out(y)


class _CrossAttention(nn.Module):
    """Cross attention from latent tokens to a text embedding sequence."""

    def __init__(self, dim: int, ctx_dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        b, n, _ = x.shape
        m = ctx.shape[1]
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(b, m, 2, self.heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out(y)


class _RefinerBlock(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = _SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = _CrossAttention(dim, ctx_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, ctx: Tensor | None) -> Tensor:
        x = x + self.self_attn(self.norm1(x))
        if ctx is not None:
            x = x + self.cross_attn(self.norm2(x), ctx)
        x = x + self.ff(self.norm3(x))
        return x


def _sinusoidal_positions(seq_len: int, dim: int, device: torch.device) -> Tensor:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * freqs)
    pe[:, 1::2] = torch.cos(pos * freqs)
    return pe


class MeshMindRefiner(nn.Module):
    """Compact transformer that refines latent codes in-place.

    Parameters
    ----------
    in_channels:
        Channel count of the incoming latent grid (4 for SD VAE outputs,
        configurable for ShapE/triplane latents).
    dim:
        Internal hidden dimension.
    depth:
        Number of refiner blocks.
    heads:
        Attention head count.
    ctx_dim:
        Channel count of the conditioning sequence (e.g. CLIP text embeddings).
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        ctx_dim: int = 768,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.proj_in = nn.Linear(in_channels, dim)
        self.proj_out = nn.Linear(dim, in_channels)
        self.blocks = nn.ModuleList(
            [_RefinerBlock(dim=dim, ctx_dim=ctx_dim, heads=heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        latents: Tensor,
        context: Tensor | None = None,
        strength: float = 1.0,
    ) -> Tensor:
        """Refine a latent tensor.

        ``latents`` may be either a 4D grid ``(B, C, H, W)`` (image latents) or
        a 3D sequence ``(B, N, C)`` (token latents).  The output preserves the
        input shape.  When ``strength == 0`` the input is returned untouched
        which makes the refiner a no-op until trained.
        """
        if strength == 0.0:
            return latents

        if latents.dim() == 4:
            b, c, h, w = latents.shape
            tokens = latents.flatten(2).transpose(1, 2)
        elif latents.dim() == 3:
            b, n, c = latents.shape
            tokens = latents
            h = w = None
        else:
            raise ValueError(f"Unsupported latent shape {tuple(latents.shape)}")

        if c != self.in_channels:
            raise ValueError(
                f"Channel mismatch: refiner expects {self.in_channels} channels, got {c}"
            )

        x = self.proj_in(tokens)
        x = x + _sinusoidal_positions(x.shape[1], self.dim, x.device).unsqueeze(0)
        for block in self.blocks:
            x = block(x, context)
        x = self.norm(x)
        delta = self.proj_out(x)

        refined = tokens + strength * delta
        if h is not None and w is not None:
            return refined.transpose(1, 2).reshape(b, c, h, w)
        return refined

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
