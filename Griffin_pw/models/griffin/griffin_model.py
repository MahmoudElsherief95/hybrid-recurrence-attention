"""
Griffin Model Implementation — matches official RecurrentGemma repo exactly.

Reference: https://github.com/google-deepmind/recurrentgemma
Paper: Griffin arxiv 2402.19427

Architecture:
  - Blocks ALTERNATE between RECURRENT and ATTENTION: [REC, REC, ATT, REC, REC, ATT, ...]
  - Each ResidualBlock: RMSNorm → temporal_block → residual → RMSNorm → GatedMLP → residual
  - Recurrent block: two-branch (y-gate × x-Conv1D-RGLRU)
  - Attention block: causal local multi-head attention with sliding window
  - No positional embeddings (position encoded via RoPE in attention, implicit in recurrence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math
import itertools

from .gated_linear_recurrence import RMSNorm, RecurrentBlock, GatedMLP
from .local_attention import LocalAttentionBlock


# Block type constants — matching official repo common.py
RECURRENT = "RECURRENT"
ATTENTION  = "ATTENTION"


def _make_griffin_block_types(num_layers: int) -> List[str]:
    """Griffin pattern: [REC, REC, ATT] cycling — matches official repo."""
    pattern = itertools.cycle([RECURRENT, RECURRENT, ATTENTION])
    return list(itertools.islice(pattern, num_layers))


class ResidualBlock(nn.Module):
    """
    Single Griffin residual block — either RECURRENT or ATTENTION type.
    Matches official repo modules.py ResidualBlock exactly.

    Structure:
        raw_x  → temporal_pre_norm → temporal_block → + raw_x
               → channel_pre_norm  → GatedMLP         → + residual
    """
    def __init__(self, width: int, num_heads: int, local_window: int,
                 block_type: str, mlp_expanded_width: Optional[int] = None):
        super().__init__()
        self.block_type = block_type
        self.temporal_pre_norm = RMSNorm(width)
        self.channel_pre_norm = RMSNorm(width)
        self.mlp = GatedMLP(width, mlp_expanded_width or width * 4)

        if block_type == RECURRENT:
            self.temporal_block = RecurrentBlock(width, num_heads)
        else:
            self.temporal_block = LocalAttentionBlock(
                d_model=width, num_heads=num_heads, local_window=local_window
            )

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw_x = x
        # Temporal mixing (recurrent or attention)
        normed = self.temporal_pre_norm(raw_x)
        if self.block_type == RECURRENT:
            temporal_out, new_h = self.temporal_block(normed, hidden_state)
        else:
            temporal_out, new_h = self.temporal_block(normed), None

        residual = temporal_out + raw_x
        # Channel mixing (gated MLP)
        out = self.mlp(self.channel_pre_norm(residual)) + residual
        return out, new_h


class GriffinModel(nn.Module):
    """
    Griffin Model — exact architecture from the paper and official repo.

    Block pattern: [RECURRENT, RECURRENT, ATTENTION, RECURRENT, RECURRENT, ATTENTION, ...]
    No positional embeddings — position is handled implicitly.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        self.num_layers = config["num_layers"]

        block_types = _make_griffin_block_types(self.num_layers)
        num_heads  = config.get("num_heads", 8)
        local_win  = config.get("local_window", 64)
        mlp_width  = config.get("mlp_expanded_width", d_model * 4)

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=math.sqrt(1.0 / d_model))

        self.blocks = nn.ModuleList([
            ResidualBlock(
                width=d_model, num_heads=num_heads,
                local_window=local_win, block_type=bt,
                mlp_expanded_width=mlp_width,
            )
            for bt in block_types
        ])
        self.output_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie weights

    def forward(self, input_ids, attention_mask=None, hidden_states=None,
                output_hidden_states=False, return_dict=True):
        # Embed tokens — no positional embeddings (position implicit in recurrence + RoPE in attention)
        x = self.token_embedding(input_ids)

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        all_hidden_states, new_hidden_states = [], []
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            x, block_hidden = block(x, hidden_states[i])
            new_hidden_states.append(block_hidden)

        x = self.output_norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        logits = self.lm_head(x)
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': all_hidden_states if output_hidden_states else None,
                'new_hidden_states': new_hidden_states,
            }
        return logits
