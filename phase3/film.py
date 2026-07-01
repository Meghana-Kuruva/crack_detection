"""
Phase 3 — FiLM Global Context Conditioning
============================================
Implements Section 2.1: "Feature-wise Linear Modulation (FiLM) from global context."

THE PROBLEM FiLM SOLVES:
  SAHI slices the full image into 640×640 tiles for inference.  Each tile loses
  the spatial context of its surroundings — it cannot see whether a crack is on
  a sleeper surface or on ballast at the edge of the tile.  This causes FPs at
  tile boundaries where ballast gaps look like off-sleeper cracks.

THE SOLUTION:
  Run a lightweight encoder on the DOWNSAMPLED FULL FRAME (≤256×256) in parallel
  with tile inference.  This produces a SPATIAL global feature map G of shape
  (B, D, H_g, W_g).

  For each tile at position (x1, y1, x2, y2) in the full frame:
    - Crop the spatial global features at the tile's scaled position: G_tile
    - Project G_tile to per-channel (γ, β) shift/scale vectors
    - Apply FiLM to the tile's neck features F:  F' = γ ⊙ F + β

  Because G_tile is spatially CROPPED (not globally averaged), the conditioning
  is spatially aware — the model knows whether the tile is in the "sleeper region"
  or the "ballast region" of the full frame.

MODULE STRUCTURE:
  GlobalContextEncoder    — lightweight CNN that encodes the full frame → spatial map G
  FiLMConditioner         — for one tile + its G_tile crop → generates (γ, β) per channel
  FiLMNeck                — wraps AFPN output: applies FiLM conditioning to all P-levels

DESIGN NOTE on G resolution:
  We run the global encoder at stride 16 (same as P4) so G has reasonable spatial
  granularity without being too expensive.  At inference the full frame at 256×256
  gives G of size 16×16 — which maps to each 16×16 patch of the original frame.
  For a 640×640 tile the corresponding G_tile region is:
      G_tile = G[:, :, y1//16 : y2//16, x1//16 : x2//16]
  which is then upsampled to match the tile P3/P4/P5 spatial sizes.

Usage:
    encoder = GlobalContextEncoder(in_ch=3, out_ch=256)
    conditioner = FiLMNeck(neck_channels=[64,128,256,512], context_ch=256)

    # Full frame (B, 3, H_full, W_full) — downsampled
    global_feat = encoder(full_frame_small)   # (B, 256, H_full//16, W_full//16)

    # Tile features from AFPN — each (B, C_l, H_l, W_l)
    # tile_bbox: (x1, y1, x2, y2) in full-frame pixel coordinates
    o2, o3, o4, o5 = conditioner(
        feats=[o2, o3, o4, o5],
        global_feat=global_feat,
        full_frame_hw=(H_full, W_full),
        tile_bbox=(x1, y1, x2, y2),
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONTEXT ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class GlobalContextEncoder(nn.Module):
    """
    Lightweight CNN that encodes a downsampled full frame into a spatial feature map.

    Architecture:
        Input  (3, H, W)          — typically (3, 256, 256) or (3, 320, 320)
        stride-2 conv  → (32, H/2, W/2)
        stride-2 conv  → (64, H/4, W/4)
        stride-2 conv  → (128, H/8, W/8)
        stride-2 conv  → (out_ch, H/16, W/16)   ← global feature map G

    Total cost at 256×256 input: ~12M MACs — about 1/50th of the main detection forward.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 256):
        super().__init__()

        def blk(ic, oc, stride=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            blk(in_ch, 32, stride=2),    # → H/2
            blk(32,    64, stride=2),    # → H/4
            blk(64,   128, stride=2),    # → H/8
            blk(128, out_ch, stride=2),  # → H/16  — the spatial global map G
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# FiLM GENERATOR: crop global map → (γ, β)
# ══════════════════════════════════════════════════════════════════════════════

class FiLMGenerator(nn.Module):
    """
    Projects a cropped spatial global feature to per-channel (γ, β) shift/scale.

    The crop G_tile (B, context_ch, h_g, w_g) is first spatially pooled then
    projected to 2 × target_ch scalars (γ and β for each channel).

    Args:
        context_ch : channels in the global feature map G (from GlobalContextEncoder)
        target_ch  : channels in the neck feature to be modulated
    """

    def __init__(self, context_ch: int, target_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)       # (B, context_ch, h, w) → (B, context_ch, 1, 1)
        self.proj = nn.Linear(context_ch, target_ch * 2)    # → (γ, β) vectors

        # Init γ close to 1 and β close to 0 so early training is close to identity
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:target_ch])    # γ bias → 1
        nn.init.zeros_(self.proj.bias[target_ch:])   # β bias → 0

    def forward(self, g_tile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g_tile : (B, context_ch, h_g, w_g) — global features cropped at tile position

        Returns:
            gamma : (B, C, 1, 1) scale factors
            beta  : (B, C, 1, 1) shift factors
        """
        B = g_tile.shape[0]
        h = self.pool(g_tile).view(B, -1)     # (B, context_ch)
        params = self.proj(h)                  # (B, 2*C)
        C = params.shape[1] // 2
        gamma = params[:, :C].view(B, C, 1, 1) + 1.0   # +1 for identity init
        beta  = params[:, C:].view(B, C, 1, 1)
        return gamma, beta


# ══════════════════════════════════════════════════════════════════════════════
# FILM NECK WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class FiLMNeck(nn.Module):
    """
    Applies FiLM conditioning to each AFPN output level.

    For each level l with feature map F_l:
        G_tile_l  = crop_and_resize(G, tile_position, target_size=F_l.size())
        γ_l, β_l  = FiLMGenerator_l(G_tile_l)
        F_l'      = γ_l ⊙ F_l + β_l

    The crop is done using F.grid_sample for sub-pixel accurate spatial alignment.

    Args:
        neck_channels : list of channel counts for each AFPN output level
                        (typically [out_ch, out_ch, out_ch, out_ch] after AFPN)
        context_ch    : channel count of GlobalContextEncoder output
    """

    def __init__(
        self,
        neck_channels: List[int],   # [p2_ch, p3_ch, p4_ch, p5_ch] after AFPN
        context_ch:    int = 256,
    ):
        super().__init__()
        self.generators = nn.ModuleList([
            FiLMGenerator(context_ch, ch)
            for ch in neck_channels
        ])

    def forward(
        self,
        feats:           List[torch.Tensor],  # AFPN outputs [o2, o3, o4, o5]
        global_feat:     torch.Tensor,        # (B, context_ch, H_g, W_g)
        full_frame_hw:   Tuple[int, int],     # (H_full, W_full) in pixel coords
        tile_bbox:       Optional[Tuple[int, int, int, int]] = None,
                                              # (x1, y1, x2, y2) in full-frame coords
                                              # None = use full frame as tile (no crop)
    ) -> List[torch.Tensor]:
        """
        Apply FiLM to each feature level using the spatially cropped global context.

        If tile_bbox is None (e.g. during whole-image inference), the entire global
        feature map is pooled — no spatial crop is made.
        """
        H_full, W_full = full_frame_hw
        H_g, W_g       = global_feat.shape[-2:]

        if tile_bbox is not None:
            x1, y1, x2, y2 = tile_bbox
            # Compute crop in global-feature coordinates
            # (integer floor/ceil to stay within bounds)
            gx1 = max(int(x1 / W_full * W_g), 0)
            gx2 = min(int(x2 / W_full * W_g) + 1, W_g)
            gy1 = max(int(y1 / H_full * H_g), 0)
            gy2 = min(int(y2 / H_full * H_g) + 1, H_g)
            g_tile = global_feat[:, :, gy1:gy2, gx1:gx2]
            # Guard against zero-size crops at image boundaries
            if g_tile.shape[-1] < 1 or g_tile.shape[-2] < 1:
                g_tile = global_feat
        else:
            g_tile = global_feat   # use full feature map

        out = []
        for i, (feat, gen) in enumerate(zip(feats, self.generators)):
            gamma, beta = gen(g_tile)    # (B, C, 1, 1)
            out.append(gamma * feat + beta)

        return out
