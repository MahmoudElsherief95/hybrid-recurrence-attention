"""
Gated Linear Recurrence implementation for Griffin/Hawk models
Based on the Griffin paper: "Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GatedLinearRecurrence(nn.Module):
    """
    Gated Linear Recurrence layer as described in the Griffin paper.
    
    This implements the core recurrent component that processes sequences
    with gated linear recurrences for efficient long-range modeling.
    """
    
    def __init__(
        self,
        hidden_size: int,
        gate_type: str = "glu",
        activation: str = "swish",
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.activation = activation
        
        # Linear transformations for input processing
        self.input_proj = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        
        # Recurrent state transformation
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Gating mechanism
        if gate_type == "glu":
            self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
        elif gate_type == "simple":
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
            
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate scales"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        if name == "swish":
            return F.silu
        elif name == "gelu":
            return F.gelu
        elif name == "relu":
            return F.relu
        elif name == "tanh":
            return F.tanh
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Gated Linear Recurrence layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            hidden_state: Previous hidden state of shape (batch_size, hidden_size)
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size)
            final_hidden: Final hidden state of shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.hidden_size, 
                dtype=x.dtype, device=x.device
            )
        
        # Process input
        input_features = self.input_proj(x)  # (B, L, 3*H)
        u, v, w = input_features.chunk(3, dim=-1)  # Each: (B, L, H)
        
        # Apply activation to input features
        activation_fn = self._get_activation(self.activation)
        u = activation_fn(u)
        v = activation_fn(v)
        
        # Process sequence step by step
        outputs = []
        current_hidden = hidden_state
        
        for t in range(seq_len):
            # Current input at time t
            u_t = u[:, t, :]  # (B, H)
            v_t = v[:, t, :]  # (B, H)
            w_t = w[:, t, :]  # (B, H)
            
            # Gating mechanism
            if self.gate_type == "glu":
                gate_features = self.gate_proj(current_hidden)  # (B, 2*H)
                gate_values, gate_transforms = gate_features.chunk(2, dim=-1)
                gate = torch.sigmoid(gate_values) * activation_fn(gate_transforms)
            else:  # simple gate
                gate = torch.sigmoid(self.gate_proj(current_hidden))
            
            # Recurrent update
            recurrent_input = self.recurrent_proj(current_hidden)
            
            # Combine inputs with gating
            new_hidden = gate * recurrent_input + (1 - gate) * (u_t * v_t + w_t)
            
            # Apply dropout
            new_hidden = self.dropout(new_hidden)
            
            # Store output
            output_t = self.output_proj(new_hidden)
            outputs.append(output_t)
            
            # Update hidden state
            current_hidden = new_hidden
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (B, L, H)
        
        return output, current_hidden


class MultiLayerGatedRecurrence(nn.Module):
    """
    Multi-layer Gated Linear Recurrence module.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        gate_type: str = "glu",
        activation: str = "swish",
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        
        # Recurrence layers
        self.layers = nn.ModuleList([
            GatedLinearRecurrence(
                hidden_size=hidden_size,
                gate_type=gate_type,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size)
                for _ in range(num_layers)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through multiple recurrence layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            hidden_states: List of hidden states for each layer
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size)
            final_hidden_states: List of final hidden states for each layer
        """
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        current_input = x
        final_hidden_states = []
        
        for i, layer in enumerate(self.layers):
            # Apply recurrence layer
            layer_output, layer_hidden = layer(current_input, hidden_states[i])
            
            # Apply layer normalization if enabled
            if self.layer_norm:
                layer_output = self.layer_norms[i](layer_output)
            
            # Residual connection
            if current_input.shape == layer_output.shape:
                current_input = current_input + layer_output
            else:
                current_input = layer_output
            
            final_hidden_states.append(layer_hidden)
        
        return current_input, final_hidden_states
