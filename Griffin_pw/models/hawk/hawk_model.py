"""
Hawk Model Implementation  matches official RecurrentGemma repo exactly.

Reference: https://github.com/google-deepmind/recurrentgemma
Paper: Griffin arxiv 2402.19427 (Hawk = all-RECURRENT variant)

Architecture:
  - ALL blocks are RECURRENT (no attention at all)
  - Each block: RMSNorm -> RecurrentBlock -> residual -> RMSNorm -> GatedMLP -> residual
  - RecurrentBlock: two-branch (y-gate x x-Conv1D-RGLRU)
  - No positional embeddings -- position implicit in recurrence order
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffin.gated_linear_recurrence import RMSNorm, RecurrentBlock, GatedMLP


class HawkResidualBlock(nn.Module):
    """Single Hawk residual block -- all-recurrent version of Griffin's ResidualBlock."""
    def __init__(self, width: int, num_heads: int, mlp_expanded_width: Optional[int] = None):
        super().__init__()
        self.temporal_pre_norm = RMSNorm(width)
        self.temporal_block = RecurrentBlock(width, num_heads)
        self.channel_pre_norm = RMSNorm(width)
        self.mlp = GatedMLP(width, mlp_expanded_width or width * 4)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_x = x
        normed = self.temporal_pre_norm(raw_x)
        temporal_out, new_h = self.temporal_block(normed, hidden_state)
        residual = temporal_out + raw_x
        out = self.mlp(self.channel_pre_norm(residual)) + residual
        return out, new_h


class HawkModel(nn.Module):
    """
    Hawk Model -- pure-recurrent variant (all blocks are RG-LRU, no attention).
    Matches official repo Preset.HAWK_PAPER_7B but scaled to small config.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        num_heads = config.get("num_heads", 8)
        mlp_width = config.get("mlp_expanded_width", d_model * 4)

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=math.sqrt(1.0 / d_model))

        self.blocks = nn.ModuleList([
            HawkResidualBlock(d_model, num_heads, mlp_width)
            for _ in range(self.num_layers)
        ])
        self.output_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie weights

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states=None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        # No positional embeddings -- matches official Hawk
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
