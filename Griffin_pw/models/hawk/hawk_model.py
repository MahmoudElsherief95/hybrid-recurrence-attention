"""
Hawk Model Implementation
Pure RNN with Gated Linear Recurrences (recurrence-only component of Griffin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class GatedRecurrenceBlock(nn.Module):
    """Gated Recurrence block for Hawk model."""
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["d_model"]
        gate_type = config.get("gate_type", "glu")
        activation = config.get("activation", "swish")
        dropout = config.get("dropout", 0.1)
        bias = config.get("bias", True)
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.activation = activation
        self.input_proj = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2 if gate_type == "glu" else hidden_size, bias=bias)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_activation(self, name: str):
        """Get activation function"""
        activations = {
            "swish": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
            "tanh": F.tanh
        }
        return activations.get(name, F.silu)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of gated recurrence block.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            hidden_state: Previous hidden state (batch_size, hidden_size)
            
        Returns:
            output: Output tensor (batch_size, seq_len, hidden_size)
            final_hidden: Final hidden state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.hidden_size,
                dtype=x.dtype, device=x.device
            )
        
        # Process input
        input_features = self.input_proj(x)
        u, v, w = input_features.chunk(3, dim=-1)
        
        activation_fn = self._get_activation(self.activation)
        u = activation_fn(u)
        v = activation_fn(v)
        
        # Recurrent processing
        outputs = []
        current_hidden = hidden_state
        
        for t in range(seq_len):
            # Current inputs
            u_t = u[:, t, :]
            v_t = v[:, t, :]
            w_t = w[:, t, :]
            
            # Gating
            if self.gate_type == "glu":
                gate_features = self.gate_proj(current_hidden)
                gate_values, gate_transforms = gate_features.chunk(2, dim=-1)
                gate = torch.sigmoid(gate_values) * activation_fn(gate_transforms)
            else:
                gate = torch.sigmoid(self.gate_proj(current_hidden))
            
            # Recurrent update
            recurrent_input = self.recurrent_proj(current_hidden)
            new_hidden = gate * recurrent_input + (1 - gate) * (u_t * v_t + w_t)
            
            # Apply dropout and layer norm
            new_hidden = self.dropout(new_hidden)
            new_hidden = self.layer_norm(new_hidden)
            
            # Output projection
            output_t = self.output_proj(new_hidden)
            outputs.append(output_t)
            
            current_hidden = new_hidden
        
        output = torch.stack(outputs, dim=1)
        return output, current_hidden


class HawkModel(nn.Module):
    """Hawk Model - Pure RNN with Gated Linear Recurrences."""
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            GatedRecurrenceBlock(config) for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(config.get("dropout", 0.1)),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(config.get("dropout", 0.1))
            )
            for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding.weight, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[list] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass of Hawk model.
        
        Args:
            input_ids: Input token ids (batch_size, seq_len)
            hidden_states: Previous hidden states
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dictionary
            
        Returns:
            Dictionary with logits and optional hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.positional_embedding(position_ids)
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Initialize hidden states
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        # Process through recurrence blocks
        all_hidden_states = []
        new_hidden_states = []
        
        for i, (block, ffn, layer_norm) in enumerate(
            zip(self.blocks, self.ffn_layers, self.layer_norms)
        ):
            if output_hidden_states:
                all_hidden_states.append(x)
            
            # Recurrence block with residual connection
            residual = x
            x, block_hidden = block(x, hidden_states[i])
            x = residual + x
            x = layer_norm(x)
            
            # Feed-forward with residual connection
            residual = x
            x = residual + ffn(x)
            
            new_hidden_states.append(block_hidden)
        
        # Final layer norm
        x = self.output_norm(x)
        
        if output_hidden_states:
            all_hidden_states.append(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': all_hidden_states if output_hidden_states else None,
                'new_hidden_states': new_hidden_states
            }
        else:
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
        Generate text using Hawk model.
        
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
        
        hidden_states = None
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                outputs = self.forward(
                    generated,
                    hidden_states=hidden_states,
                    return_dict=True
                )
                
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                hidden_states = outputs['new_hidden_states']
                
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
