"""
MaskNet: Adversarial Frequency Mask Generator for Range-Doppler Heatmaps
=========================================================================
Implements f_mask(Z_rd; Phi_mask) -> M_mask in [0,1]^{2 x N_chip x N_block}

Key constraint: the zero-Doppler column (vel = 0) carries a strong DC /
static-clutter component visible as a bright vertical stripe in P_rd.
The mask MUST suppress this column so the adversarial mixup never distorts it.

IMPORTANT — vel_axis alignment
  The zero-Doppler bin is at  argmin(|vel_axis|),  which equals N_block // 2
  ONLY when vel_axis is perfectly symmetric.  MaskNet and ZeroDopplerBias
  accept a `vel_axis` array and derive v0 from it, mirroring plot_rd():

      zero_idx = np.argmin(np.abs(vel_axis))        # same as plot_rd
      gate[:, zero_idx : zero_idx + 2] = 0          # 2-col wide, same as plot

Two enforcement layers:
  1. ZeroDopplerBias  — learnable Gaussian dip in logit space, centred at v0.
  2. Hard gate        — fixed buffer, zeros columns [v0, v0+1] after sigmoid.

Architecture: lightweight U-Net encoder-decoder with:
  - Doppler-aware positional encoding injected at the encoder input
  - Column-wise attention in the bottleneck  (captures full Doppler structure)
  - Squeeze-and-excitation in each decoder stage
  - ZeroDopplerBias head before final sigmoid
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.isac import N_chip, N_block, vel_axis


# ============================================================================
# 1.  Positional / Structural Encodings
# ============================================================================

class DopplerPositionalEncoding(nn.Module):
    """
    Injects a learnable 1-D sinusoidal Doppler coordinate into every spatial
    location.  Shape: (1, 1, 1, N_block) broadcast over (B, C, N_chip, N_block).

    The sinusoid is centred at v0 = argmin(|vel_axis|) so the encoding's
    zero-crossing aligns with the actual zero-Doppler bin, not necessarily
    with the array midpoint N_block // 2.
    """

    def __init__(self, n_block: int, v0: int, out_ch: int = 1):
        """
        Parameters
        ----------
        n_block : number of Doppler bins
        v0     : zero-Doppler bin index  (argmin(|vel_axis|))
        out_ch : output channels after the 1x1 projection
        """
        super().__init__()
        # Coordinate relative to the true zero-Doppler bin
        idx   = torch.arange(n_block, dtype=torch.float32) - v0
        # Normalise so the full axis spans [-pi, pi] around v0
        scale = math.pi / max(v0, n_block - 1 - v0, 1)
        pos   = idx * scale                                       # (N_block,)
        base  = torch.stack([torch.cos(pos), torch.sin(pos)], 0) # (2, N_block)
        self.register_buffer("base", base.unsqueeze(0).unsqueeze(2))  # (1,2,1,N)

        # Small learnable mixer
        self.proj = nn.Conv2d(2, out_ch, kernel_size=1, bias=True)

    def forward(self, H: int) -> torch.Tensor:
        """Returns (1, out_ch, H, N_block) — broadcast over B and range."""
        enc = self.base.expand(1, -1, H, -1)   # (1, 2, H, N_block)
        return self.proj(enc)                   # (1, out_ch, H, N_block)


# ============================================================================
# 2.  Zero-Doppler Bias  (learnable suppression profile)
# ============================================================================

class ZeroDopplerBias(nn.Module):
    """
    Learnable 1-D profile over the Doppler axis added to CNN logits
    before sigmoid.  Initialised as a sharp Gaussian dip at v0 so that
    sigmoid(logit + bias) ~= 0 at the zero-Doppler bin from iteration 1.

    v0 is derived from vel_axis as argmin(|vel_axis|), matching plot_rd().
    Both the depth and width (sigma) are learnable so the network can adapt
    the suppression zone during training.
    """

    def __init__(self, n_block: int, v0: int,
                 init_depth: float = -6.0,
                 init_sigma_frac: float = 0.04):
        """
        Parameters
        ----------
        n_block          : number of Doppler bins
        v0              : zero-Doppler bin index  (argmin(|vel_axis|))
        init_depth      : initial logit dip depth (large negative -> ~0 mask)
        init_sigma_frac : initial Gaussian half-width as fraction of n_block
        """
        super().__init__()
        self.n_block = n_block

        # Learnable parameters
        self.depth = nn.Parameter(torch.tensor(float(init_depth)))
        self.sigma = nn.Parameter(torch.tensor(init_sigma_frac * n_block))

        # Coordinate vector centred on the TRUE zero-Doppler bin
        v = torch.arange(n_block, dtype=torch.float32) - v0
        self.register_buffer("v_coord", v)                 # (N_block,)

    def forward(self) -> torch.Tensor:
        """
        Returns (1, 1, 1, N_block) bias.

        Profile:  depth * exp(-0.5 * (v / sigma)^2),   depth <= 0  (clamped)
        """
        sigma_safe = self.sigma.abs().clamp(min=1.0)
        dip = self.depth.clamp(max=0.0) * torch.exp(
            -0.5 * (self.v_coord / sigma_safe) ** 2)
        return dip.view(1, 1, 1, self.n_block)             # broadcastable


# ============================================================================
# 3.  Building Blocks
# ============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ColumnWiseAttention(nn.Module):
    """
    Global column attention: collapses the range axis (H) via avg-pool,
    processes all N_block columns with a 1-D conv, then broadcasts back.

    This lets the bottleneck reason about the full Doppler spectrum at once —
    critical for detecting and suppressing the zero-Doppler stripe.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d((1, None))    # (B, C, 1, W)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=5,
                      padding=2, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=5,
                      padding=2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x).squeeze(2)          # (B, C, W)
        w   = self.conv(gap).unsqueeze(2)     # (B, C, 1, W)
        return x * w                          # (B, C, H, W)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * w


class EncoderBlock(nn.Module):
    """Conv x 2 -> skip; MaxPool -> downsampled."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv2(self.conv1(x))
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """Bilinear up -> concat skip -> Conv x 2 -> SE."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=False)
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.se    = ChannelAttention(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return self.se(self.conv2(self.conv1(torch.cat([x, skip], dim=1))))


# ============================================================================
# 4.  MaskNet
# ============================================================================

class MaskNet(nn.Module):
    """
    f_mask(Z_rd; Phi_mask) -> M_mask in [0,1]^{B x 2 x N_chip x N_block}

    Zero-Doppler constraint
    -----------------------
    The zero-Doppler bin is located at

        v0 = argmin(|vel_axis|)          # mirrors your plot_rd() logic exactly

    which is NOT necessarily N_block // 2 for asymmetric velocity axes.

    The hard gate zeros columns [v0, v0+1] (2 columns wide), identical to:

        P_rd_display[:, zero_idx : zero_idx + 2] = np.nan   # in plot_rd()

    Two enforcement layers:
      Layer 1 - ZeroDopplerBias: learnable Gaussian dip in logit space at v0.
      Layer 2 - Hard gate: fixed buffer, zeros [v0, v0+1] after sigmoid.

    Args
    ----
    vel_axis : 1-D array of Doppler velocities, length N_block.
               Accepts numpy ndarray or torch.Tensor.
               n_block = len(vel_axis) and v0 = argmin(|vel_axis|) are both
               derived from this — no need to pass them separately.
    n_chip   : range bins (N_chip from configs.isac, default)
    in_ch    : input channels (default 2: real + imag)
    base_ch  : feature-map width at first encoder stage (default 16)
    depth    : number of encoder/decoder stages (default 3)
    """

    def __init__(self,
                 n_chip:  int = N_chip,
                 n_block: int = N_block,
                 in_ch:   int = 2,
                 base_ch: int = 16,
                 depth:   int = 3):
        super().__init__()

        self.depth  = depth
        out_ch      = 2

        # ── Everything Doppler-related comes from vel_axis alone ─────────────
        if isinstance(vel_axis, torch.Tensor):
            vel_np = vel_axis.cpu().numpy()
        else:
            vel_np = np.asarray(vel_axis, dtype=float)
        v0          = int(np.argmin(np.abs(vel_np))) # zero-Doppler bin index
        self.n_chip = n_chip 
        self.n_block = n_block
        self.v0     = v0

        # ── Doppler positional encoding (centred at true v0) ─────────────────
        self.dpe = DopplerPositionalEncoding(n_block, v0=v0, out_ch=1)

        # Input: 2 complex channels + 1 positional channel
        first_in = in_ch + 1

        # ── Channel sizes ────────────────────────────────────────────────────
        chs = [first_in] + [base_ch * (2 ** i) for i in range(depth)]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList(
            [EncoderBlock(chs[i], chs[i + 1]) for i in range(depth)]
        )

        # ── Bottleneck (column-wise attention for Doppler structure) ─────────
        btn = chs[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBNReLU(chs[-1], btn),
            DepthwiseSeparableConv(btn, btn),
            ColumnWiseAttention(btn),      # global Doppler reasoning
            ChannelAttention(btn),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_in = [btn] + [chs[depth - i] for i in range(depth - 1)]
        dec_sk = [chs[depth - i]         for i in range(depth)]
        dec_ou = [chs[depth - i]         for i in range(depth)]
        self.decoders = nn.ModuleList(
            [DecoderBlock(dec_in[i], dec_sk[i], dec_ou[i])
             for i in range(depth)]
        )

        # ── Output head: logits (no activation yet) ──────────────────────────
        self.head_logit = nn.Conv2d(dec_ou[-1], out_ch, kernel_size=1)

        # ── Layer 1: learnable zero-Doppler bias in logit space ───────────────
        self.zd_bias = ZeroDopplerBias(n_block, v0=v0)

        # ── Layer 2: hard gate — zeros [v0, v0+1], matching plot_rd NaN mask ──
        #    plot_rd:  P_rd_display[:, zero_idx : zero_idx + 2] = np.nan
        gate = torch.ones(1, 1, 1, n_block)
        gate[:, :, :, v0 : v0 + 2] = 0.0
        self.register_buffer("zd_gate", gate)  # (1, 1, 1, N_block)

    # -------------------------------------------------------------------------
    def forward(self, Z_rd: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        Z_rd : (B, 2, N_chip, N_block)   noisy or denoised RD map

        Returns
        -------
        M_mask : (B, 2, N_chip, N_block)  in [0, 1],
                 with M_mask[:, :, :, v0:v0+2] == 0  (mirrors plot_rd NaN mask)
        """
        B, _, H, W = Z_rd.shape

        # 1. Inject Doppler positional encoding
        dpe = self.dpe(H).expand(B, -1, -1, -1)   # (B, 1, H, W)
        x   = torch.cat([Z_rd, dpe], dim=1)        # (B, 3, H, W)

        # 2. Encoder
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # 3. Bottleneck
        x = self.bottleneck(x)

        # 4. Decoder
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[self.depth - 1 - i])

        # 5. Logit head
        logits = self.head_logit(x)                # (B, 2, H, W)

        # 6. Add learnable zero-Doppler dip to logits
        logits = logits + self.zd_bias()           # broadcast (1,1,1,W)

        # 7. Sigmoid -> mask in (0, 1)
        M_mask = torch.sigmoid(logits)             # (B, 2, H, W)

        # 8. Hard gate: unconditionally zero [v0, v0+1]
        M_mask = M_mask * self.zd_gate             # (B, 2, H, W)

        return M_mask


if __name__ == "__main__":
    torch.manual_seed(42)

    B = 4

    # vel_axis, N_chip, N_block all come from configs.isac (imported above)
    net    = MaskNet(in_ch=2)          # n_chip defaults to N_chip
    Z_rd   = torch.randn(B, 2, N_chip, N_block)
    M_mask = net(Z_rd)

    v0 = net.v0
    print(f"N_chip={net.n_chip}, N_block={net.n_block}  (n_block=len(vel_axis))")
    print(f"v0={v0}  (argmin(|vel_axis|))")
    print(f"Input  Z_rd  : {tuple(Z_rd.shape)}")
    print(f"Output M_mask: {tuple(M_mask.shape)}")
    print(f"Gate cols [{v0},{v0+1}] max: {M_mask[:,:,:,v0:v0+2].max():.6f}  [must be 0]")
    print(f"Params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")