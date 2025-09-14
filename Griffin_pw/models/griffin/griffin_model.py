"""
Griffin Model Implementation
Hybrid model combining Gated Linear Recurrences with Local Attention
Based on the official RecurrentGemma implementation from Google DeepMind

Reference: https://github.com/google-deepmind/recurrentgemma
Paper: https://arxiv.org/abs/2402.19427
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .gated_linear_recurrence import MultiLayerGatedRecurrence
from .local_attention import MultiHeadLocalAttention


class GriffinBlock(nn.Module):
    """
    Single Griffin block combining recurrence and attention.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        self.mixing_alpha = config.get("mixing_alpha", 0.5)
        self.layer_norm = config.get("layer_norm", True)
        self.recurrence = MultiLayerGatedRecurrence(
            hidden_size=d_model,
            num_layers=1,
            gate_type=config.get("gate_type", "glu"),
            activation=config.get("activation", "swish"),
            dropout=config.get("dropout", 0.1),
            layer_norm=False
        )
        self.attention = MultiHeadLocalAttention(
            d_model=d_model,
            num_heads=config["num_heads"],
            num_layers=1,
            local_window=config.get("local_window", 256),
            dropout=config.get("dropout", 0.1),
            layer_norm=False
        )
        if self.layer_norm:
            self.recurrence_norm = nn.LayerNorm(d_model)
            self.attention_norm = nn.LayerNorm(d_model)
            self.output_norm = nn.LayerNorm(d_model)
        self.recurrence_proj = nn.Linear(d_model, d_model)
        self.attention_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.get("dropout", 0.1))
        )
        if self.layer_norm:
            self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, hidden_states=None, attention_mask=None):
        residual = x
        recurrence_input = self.recurrence_norm(x) if self.layer_norm else x
        recurrence_output, new_hidden_states = self.recurrence(recurrence_input, hidden_states)
        recurrence_output = self.recurrence_proj(recurrence_output)
        attention_input = self.attention_norm(x) if self.layer_norm else x
        attention_output, _ = self.attention(attention_input, attention_mask)
        attention_output = self.attention_proj(attention_output)
        mixed_output = self.mixing_alpha * recurrence_output + (1 - self.mixing_alpha) * attention_output
        x = residual + mixed_output
        ffn_input = self.ffn_norm(x) if self.layer_norm else x
        ffn_output = self.ffn(ffn_input)
        output = x + ffn_output
        if self.layer_norm:
            output = self.output_norm(output)
        return output, new_hidden_states


class GriffinModel(nn.Module):
    """
    Complete Griffin Model implementation.
    
    This model combines gated linear recurrences with local attention
    for efficient sequence modeling.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GriffinBlock(config) for _ in range(self.num_layers)
        ])
        if config.get("layer_norm", True):
            self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate scales"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, hidden_states=None, output_hidden_states=False, return_dict=True):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.positional_embedding(position_ids)
        x = self.dropout(x)
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        all_hidden_states, new_hidden_states = [], []
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            x, block_hidden = block(x, hidden_states[i], attention_mask)
            new_hidden_states.append(block_hidden)
        if hasattr(self, 'output_norm'):
            x = self.output_norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        logits = self.lm_head(x)
        if return_dict:
            return {'logits': logits, 'hidden_states': all_hidden_states if output_hidden_states else None, 'new_hidden_states': new_hidden_states}
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9, do_sample=True):
        self.eval()
        batch_size, current_length = input_ids.shape
        hidden_states = None
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_length - current_length):
                outputs = self.forward(generated, hidden_states=hidden_states, return_dict=True)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                hidden_states = outputs['new_hidden_states']
                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        return generated
