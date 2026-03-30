"""
Local Attention Model Implementation
Pure attention-based baseline with causal local sliding-window attention.
Uses RMSNorm (matching Griffin/Hawk) for a fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffin.gated_linear_recurrence import RMSNorm, GatedMLP
from griffin.local_attention import RotaryEmbedding, _rotate_half


class LocalAttentionBlock(nn.Module):
    """Local Attention block with feed-forward network."""
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        local_window = config.get("local_window", 256)
        dropout = config.get("dropout", 0.1)
        bias = config.get("bias", True)
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window = local_window
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        assert d_model % num_heads == 0
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ffn = GatedMLP(d_model, d_model * 4)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.0)  # official default: no attention dropout
        self.rotary = RotaryEmbedding(self.head_dim)
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # Causal (backward-only): token i can only see tokens i-window+1 .. i
            start = max(0, i - self.local_window + 1)
            end = i + 1
            mask[i, start:end] = True
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of local attention block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            output: Output tensor (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual connection
        residual = x
        x = self.attention_norm(x)
        
        # Multi-head attention computation
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary(seq_len, x.device)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply local mask
        local_mask = self._create_local_mask(seq_len, x.device)
        scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Attention weights and output
        # nan_to_num handles rows that are entirely masked (all -inf → 0 after softmax)
        attention_weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        # Add residual connection
        x = residual + out
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        
        return x, attention_weights


class LocalAttentionModel(nn.Module):
    """Local Attention Model — pure causal sliding-window attention baseline."""
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.local_window = config.get("local_window", 64)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=(1.0 / d_model) ** 0.5)
        self.blocks = nn.ModuleList([
            LocalAttentionBlock(config) for _ in range(self.num_layers)
        ])
        self.output_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass of Local Attention model.
        
        Args:
            input_ids: Input token ids (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return dictionary
            
        Returns:
            Dictionary with logits and optional hidden states/attentions
        """
        # No positional embeddings — matches Griffin/Hawk treatment
        x = self.token_embedding(input_ids)

        all_hidden_states, all_attentions = [], []
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(x)
            x, attention_weights = block(x, attention_mask)
            if output_attentions:
                all_attentions.append(attention_weights)

        x = self.output_norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        logits = self.lm_head(x)
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': all_hidden_states if output_hidden_states else None,
                'attentions': all_attentions if output_attentions else None,
            }
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text using Local Attention model.
        
        Args:
            input_ids: Starting tokens (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated tokens (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                outputs = self.forward(generated, return_dict=True)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
