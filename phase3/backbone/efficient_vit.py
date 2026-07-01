"""
Phase 3 — EfficientViT-style Linear-Attention Stages
======================================================
Implements the "later stages" of the hybrid backbone where long-range context
is needed for crack-vs-ballast discrimination.

KEY IDEA — why linear attention for high-res feature maps:
  Standard softmax self-attention is O(N²·d) where N = H×W.  For a 640×640
  image at stride 8, N = 6400.  That's 6400² = 40M multiplies per head — too
  expensive to run inside the neck on a T4.

  EfficientViT's insight: replace softmax with ReLU (or ELU+1) so the
  denominator K^T·1 can be pre-computed once, turning the attention into:
      O_i = Q_i · (K^T V) / (Q_i · (K^T 1))   — O(N·d²) instead of O(N²·d)
  This is the "kernel trick" for attention: when the kernel is decomposable
  (k(q,k) = φ(q)·φ(k)), the attention reduces to two matrix products.

  Reference: Liu et al., "EfficientViT: Memory Efficient Vision Transformer
  with Cascaded Group Attention", CVPR 2023.

Architecture of one EfficientViTBlock:
    x  →  DWConv(3×3, groups=C)  →  Linear attn  →  FFN(1×1 MLP)  →  x + residual

Usage:
    stage = EfficientViTStage(dim=256, depth=2, num_heads=4, head_dim=32)
    out   = stage(x)   # (B, 256, H, W) → (B, 256, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR MULTI-HEAD SELF-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class LinearMHSA(nn.Module):
    """
    Linear (ReLU-kernel) multi-head self-attention.

    Complexity: O(N · d² · num_heads) vs O(N² · d · num_heads) for softmax.
    At N=6400 (stride-8 map from a 640px image) with d=32 and 4 heads this is
    ~26M vs ~164B operations — two orders of magnitude cheaper.

    Implementation note:
        K^T V is computed first (d×d outer product), then multiplied by Q.
        Normalisation by Q·(K^T·1_N) prevents attention collapse to uniform.
    """

    def __init__(
        self,
        dim:      int,         # input/output channel dim
        num_heads: int = 4,
        head_dim:  int = 32,   # channels per head (dim need not equal num_heads*head_dim)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        inner_dim      = num_heads * head_dim

        # 1×1 projections (grouped by heads for efficiency)
        self.proj_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(inner_dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(num_heads, inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        heads, d = self.num_heads, self.head_dim
        Bh = B * heads

        # Project to Q, K, V — shape (B, 3*inner, H, W)
        qkv = self.proj_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, heads*d, H, W)

        # Apply ReLU feature map (φ(x) = ReLU(x) + ε makes kernel decomposable)
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        # Reshape to (B*heads, d, N) for batched matmul
        q = q.reshape(Bh, d, N)
        k = k.reshape(Bh, d, N)
        v = v.reshape(Bh, d, N)

        # ── Linear attention: O(N·d²) ─────────────────────────────────────────
        # KV[i,j] = Σ_n  k[i,n] · v[j,n]   → (Bh, d_k, d_v)
        kv = torch.bmm(k, v.transpose(-2, -1))      # (Bh, d, d)

        # out[j,n] = Σ_i  q[i,n] · KV[i,j] = KV^T @ q  → (Bh, d, N)
        out = torch.bmm(kv.transpose(-2, -1), q)    # (Bh, d, N)

        # Normalise: denom[n] = Σ_i  q[i,n] · (Σ_{n'} k[i,n'])
        k_sum = k.sum(dim=-1, keepdim=True)          # (Bh, d, 1)
        denom = (q * k_sum).sum(dim=1, keepdim=True) + 1e-8  # (Bh, 1, N)
        out   = out / denom                          # (Bh, d, N)

        # Reshape back to spatial (B, heads*d, H, W)
        out = out.reshape(B, heads * d, H, W)
        out = self.norm(out)
        out = self.proj_out(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# FFN (pointwise MLP — standard transformer FFN with depthwise)
# ══════════════════════════════════════════════════════════════════════════════

class EVIT_FFN(nn.Module):
    """
    MLP feedforward block: 1×1 expand → DW 3×3 → 1×1 project.
    The depthwise conv adds local spatial mixing at low cost.
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        hidden = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            # Depthwise 3×3 for local spatial mixing inside the FFN
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE EFFICIENT-VIT BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class EfficientViTBlock(nn.Module):
    """
    One EfficientViT block: DWConv → LinearMHSA → FFN, with residual connections.

    The DWConv before attention acts as a "local prior" — it captures short-range
    crack features (edge pixels) before the attention aggregates long-range context
    (crack trajectory direction, surface context for sleeper vs ballast).
    """

    def __init__(
        self,
        dim:      int,
        num_heads: int = 4,
        head_dim:  int = 32,
        ffn_expand: int = 4,
    ):
        super().__init__()

        # Local prior: DW 3×3 before attention
        self.dw_prior = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # Norms (LN approximated as GroupNorm with 1 group = LayerNorm equivalent)
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        # Linear attention
        self.attn = LinearMHSA(dim, num_heads=num_heads, head_dim=head_dim)

        # FFN
        self.ffn = EVIT_FFN(dim, expand=ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local prior branch
        x = x + self.dw_prior(x)

        # Attention branch with pre-norm
        x = x + self.attn(self.norm1(x))

        # FFN branch with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


# ══════════════════════════════════════════════════════════════════════════════
# STAGE: stack N blocks
# ══════════════════════════════════════════════════════════════════════════════

class EfficientViTStage(nn.Module):
    """
    A stage of N EfficientViT blocks at a fixed spatial resolution.

    These are inserted at the later (higher-stride) levels of the hybrid backbone
    where N (spatial tokens) is small enough that even linear attention adds up,
    but the feature resolution is still important for detecting thin cracks.
    """

    def __init__(
        self,
        dim:       int,
        depth:     int = 2,      # number of blocks
        num_heads: int = 4,
        head_dim:  int = 32,
        ffn_expand: int = 4,
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            EfficientViTBlock(dim, num_heads=num_heads, head_dim=head_dim,
                              ffn_expand=ffn_expand)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
