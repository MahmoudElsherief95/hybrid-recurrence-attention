"""
Local Attention implementation for Griffin and Local Attention models
Based on the Griffin paper's local attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LocalAttention(nn.Module):
    """
    Local Attention mechanism with sliding window approach.
    
    This implements efficient local attention that only attends to
    a fixed-size window around each position, reducing computational
    complexity from O(n²) to O(n*w) where w is the window size.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        local_window: int = 256,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window = local_window
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate scales"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create local attention mask for sliding window attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            mask: Boolean mask of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # Define window boundaries
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = True
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Local Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (B, H, L, D_head)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: (B, H, L, L)
        
        # Create local attention mask
        local_mask = self._create_local_mask(seq_len, x.device)
        
        # Apply local mask
        scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)  # (B, H, L, D_head)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final output projection
        output = self.out_proj(out)
        
        return output, attention_weights


class MultiHeadLocalAttention(nn.Module):
    """
    Multi-layer Local Attention with feed-forward networks.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        local_window: int = 256,
        ffn_hidden: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        
        if ffn_hidden is None:
            ffn_hidden = 4 * d_model
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            LocalAttention(
                d_model=d_model,
                num_heads=num_heads,
                local_window=local_window,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        if layer_norm:
            self.attention_layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
            self.ffn_layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through multiple attention layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            all_attention_weights: List of attention weights from each layer
        """
        current_input = x
        all_attention_weights = []
        
        for i in range(self.num_layers):
            # Self-attention with residual connection
            if self.layer_norm:
                normed_input = self.attention_layer_norms[i](current_input)
            else:
                normed_input = current_input
            
            attn_output, attn_weights = self.attention_layers[i](normed_input, attention_mask)
            current_input = current_input + attn_output
            all_attention_weights.append(attn_weights)
            
            # Feed-forward with residual connection
            if self.layer_norm:
                normed_input = self.ffn_layer_norms[i](current_input)
            else:
                normed_input = current_input
            
            ffn_output = self.ffn_layers[i](normed_input)
            current_input = current_input + ffn_output
        
        return current_input, all_attention_weights
