"""
RG-LRU (Real-Gated Linear Recurrent Unit) and supporting layers.

Ported from the official RecurrentGemma PyTorch implementation:
  https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/layers.py
  https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/modules.py

Key design decisions matching the paper (Griffin arxiv 2402.19427):
  - Gates come from INPUT x, not from hidden state → provably stable eigenvalues
  - `a` is a learned DIAGONAL parameter per dimension (not a full matrix)
  - `a` is initialized on a ring [min_rad, max_rad] ⊂ (0, 1) → |eigenvalues| < 1 always
  - sqrt(1 - a²) normalization preserves variance through time
  - Short Conv1D (width=4) before RG-LRU for local pattern capture
  - RecurrentBlock: two-branch design (y-gate × x-recurrence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ---------------------------------------------------------------------------
# RMSNorm (matches official repo — more stable than LayerNorm for RNNs)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Normalization — used throughout Griffin/Hawk (matches official repo)."""
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return x_normed * (self.scale + 1.0)


# ---------------------------------------------------------------------------
# BlockDiagonalLinear — block-diagonal weight matrix (matches official repo)
# Implements a grouped linear layer where `num_blocks` independent linear
# transforms operate on equal-sized slices of the hidden dimension.
# This is used for the input_gate and a_gate in RG-LRU.
# ---------------------------------------------------------------------------

class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer (no bias on gates, matching official repo)."""
    def __init__(self, width: int, num_blocks: int):
        super().__init__()
        assert width % num_blocks == 0
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        # w: (num_blocks, block_width, block_width)
        self.w = nn.Parameter(torch.empty(num_blocks, self.block_width, self.block_width))
        self.b = nn.Parameter(torch.zeros(num_blocks, self.block_width))
        self._init()

    def _init(self):
        std = math.sqrt(1.0 / self.block_width)
        nn.init.normal_(self.w, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., width)
        *leading, _ = x.shape
        x_blocks = x.view(*leading, self.num_blocks, self.block_width)
        # einsum: (..., h, i), (h, i, j) -> (..., h, j)
        y = torch.einsum('...hi,hij->...hj', x_blocks, self.w) + self.b
        return y.reshape(*leading, self.width)


# ---------------------------------------------------------------------------
# CausalConv1D — 4-token causal temporal convolution (matches official repo)
# Provides short-term local context before the RG-LRU recurrence.
# ---------------------------------------------------------------------------

class CausalConv1D(nn.Module):
    """Causal 1-D convolution with temporal_width=4 (official repo default)."""
    def __init__(self, width: int, temporal_width: int = 4):
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        # w: (temporal_width, width)  — one weight per time-step per feature
        std = math.sqrt(0.01 / temporal_width)
        self.w = nn.Parameter(torch.normal(0.0, std, size=(temporal_width, width)))
        self.b = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)  →  output: (B, L, D)
        B, L, D = x.shape
        # Left-pad with zeros so output length = input length (causal)
        pad = self.temporal_width - 1
        x_padded = F.pad(x.transpose(1, 2), (pad, 0))  # (B, D, L+pad)
        # weight for F.conv1d: (out_channels, in_channels, kernel) = (D, D, T)
        # We want depthwise: each feature independently uses its own weights
        # reshape w to (D, 1, T) for depthwise conv
        w = self.w.T.unsqueeze(1)  # (D, 1, T)
        out = F.conv1d(x_padded, w, groups=D)  # (B, D, L)
        out = out.transpose(1, 2) + self.b  # (B, L, D)
        return out


# ---------------------------------------------------------------------------
# RGLRU — Real-Gated Linear Recurrent Unit (matches official repo exactly)
#
# Paper (Griffin arxiv 2402.19427) eq. (3-4):
#   gate_x = sigmoid(W_x · x_t)    ← gate from INPUT
#   gate_a = sigmoid(W_a · x_t)    ← gate from INPUT
#   log_a  = -8 · gate_a · softplus(a_param)
#   a      = exp(log_a)             ← |a| < 1 always (stable)
#   h_t    = a ⊙ h_{t-1} + sqrt(1-a²) ⊙ (x_t · gate_x)
# ---------------------------------------------------------------------------

class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit — exact match to official repo."""
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        # Learnable diagonal A parameter — initialized on ring [0.9, 0.999]
        self.a_param = nn.Parameter(torch.empty(width))
        self.input_gate = BlockDiagonalLinear(width, num_heads)
        self.a_gate = BlockDiagonalLinear(width, num_heads)
        self._init_a_param()

    def _init_a_param(self):
        """Initialize a_param so exp(-8*sigmoid(a_param)*softplus(a_param))
        is uniform on the ring [0.9, 0.999] — matches official repo."""
        with torch.no_grad():
            # Sample uniformly on ring then apply inverse softplus transform
            unif = torch.rand_like(self.a_param)
            min_rad, max_rad = 0.9, 0.999
            a_real = 0.5 * torch.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
            # Inverse softplus: log(exp(-a_real) - 1)
            self.a_param.data = torch.log(torch.exp(-a_real) - 1.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        if hidden_state is None:
            hidden_state = torch.zeros(B, D, dtype=torch.float32, device=x.device)

        # Both gates from INPUT x (not from hidden state)
        gate_x = torch.sigmoid(self.input_gate(x))  # (B, L, D)
        gate_a = torch.sigmoid(self.a_gate(x))       # (B, L, D)

        # Compute diagonal recurrence coefficient a ∈ (0, 1) per step
        log_a = -8.0 * gate_a * F.softplus(self.a_param)  # (B, L, D)
        a = torch.exp(log_a)           # elementwise, in (0,1)
        a_square = torch.exp(2 * log_a)

        # Variance-preserving input normalization: sqrt(1 - a²)
        multiplier = torch.sqrt(torch.clamp(1.0 - a_square, min=0.0))

        # Gated + normalized input
        normalized_x = (x * gate_x) * multiplier  # (B, L, D)

        # Linear scan: h_t = a_t * h_{t-1} + normalized_x_t
        # Run in float32 for numerical stability (matches official repo)
        h = hidden_state.float()
        outputs = []
        for t in range(L):
            h = a[:, t].float() * h + normalized_x[:, t].float()
            outputs.append(h)

        y = torch.stack(outputs, dim=1).to(x.dtype)  # (B, L, D)
        return y, h.to(x.dtype)


# ---------------------------------------------------------------------------
# RecurrentBlock — full recurrent block matching official repo modules.py
#
# Two-branch design:
#   y-branch:  x → linear_y → GELU
#   x-branch:  x → linear_x → CausalConv1D → RGLRU
#   output:    linear_out(x_branch * y_branch)
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """Griffin/Hawk recurrent block — exact match to official repo."""
    def __init__(self, width: int, num_heads: int, lru_width: Optional[int] = None):
        super().__init__()
        self.lru_width = lru_width or width
        std_in = math.sqrt(1.0 / width)
        std_out = math.sqrt(2.0 / (6 * self.lru_width))  # final_w_init_variance_scale=2/num_layers approximation

        self.linear_y = nn.Linear(width, self.lru_width)
        self.linear_x = nn.Linear(width, self.lru_width)
        self.linear_out = nn.Linear(self.lru_width, width)
        self.conv1d = CausalConv1D(self.lru_width, temporal_width=4)
        self.rg_lru = RGLRU(self.lru_width, num_heads)

        nn.init.normal_(self.linear_y.weight, std=std_in)
        nn.init.zeros_(self.linear_y.bias)
        nn.init.normal_(self.linear_x.weight, std=std_in)
        nn.init.zeros_(self.linear_x.bias)
        nn.init.normal_(self.linear_out.weight, std=std_out)
        nn.init.zeros_(self.linear_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y-branch
        y = F.gelu(self.linear_y(x))
        # x-branch: conv → rg-lru
        xb = self.linear_x(x)
        xb = self.conv1d(xb)
        xb, new_h = self.rg_lru(xb, hidden_state)
        # combine
        out = self.linear_out(xb * y)
        return out, new_h


# ---------------------------------------------------------------------------
# GatedMLP — gated MLP block matching official repo MLPBlock
#   out = linear_down(GELU(up_gate) * up_act)
# ---------------------------------------------------------------------------

class GatedMLP(nn.Module):
    """Gated MLP matching official repo (two output channels from ffw_up)."""
    def __init__(self, width: int, expanded_width: Optional[int] = None):
        super().__init__()
        expanded_width = expanded_width or (width * 4)
        std = math.sqrt(1.0 / width)
        std_out = math.sqrt(2.0 / (6 * expanded_width))

        self.ffw_up = nn.Linear(width, expanded_width * 2)
        self.ffw_down = nn.Linear(expanded_width, width)

        nn.init.normal_(self.ffw_up.weight, std=std)
        nn.init.zeros_(self.ffw_up.bias)
        nn.init.normal_(self.ffw_down.weight, std=std_out)
        nn.init.zeros_(self.ffw_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.ffw_up(x)
        gate, act = up.chunk(2, dim=-1)
        return self.ffw_down(F.gelu(gate) * act)

