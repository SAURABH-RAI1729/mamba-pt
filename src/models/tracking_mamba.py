"""
MAMBA-based tracking model for particle physics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .mamba_block import ResidualMambaBlock
from .tokenizer import DetectorTokenizer


class TrackingMAMBA(nn.Module):
    """
    MAMBA-based model for particle tracking
    """
    
    def __init__(
        self,
        config: Dict,
        tokenizer: DetectorTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Model dimensions
        self.d_model = config['model']['d_model']
        self.n_layers = config['model']['n_layers']
        self.vocab_size = config['model']['vocab_size']
        
        # Embeddings
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=self.tokenizer.pad_token_id
        )
        self.position_embedding = nn.Embedding(512, self.d_model)  # Max sequence length
        
        # MAMBA layers
        mamba_kwargs = {
            'd_state': config['model']['d_state'],
            'd_conv': config['model']['d_conv'],
            'expand': config['model']['expand'],
            'dropout': config['model']['dropout'],
        }
        
        self.layers = nn.ModuleList([
            ResidualMambaBlock(self.d_model, **mamba_kwargs)
            for _ in range(self.n_layers)
        ])
        
        # Task-specific heads
        self.mdm_head = MDMHead(
            self.d_model,
            config['model']['mdm_hidden_dim'],
            self.vocab_size
        )
        
        self.ntp_head = NTPHead(
            self.d_model,
            config['model']['ntp_hidden_dim']
        )
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # Memory optimization
        self.gradient_checkpointing = config['model']['gradient_checkpointing']
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
        track_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        task: str = "mdm",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Mask for valid tokens (batch, seq_len)
            masked_positions: Positions of masked tokens for MDM task
            track_pair: Pair of tracks for NTP task
            task: "mdm" or "ntp"
        """
        outputs = {}
        
        if task == "mdm":
            if input_ids is None:
                raise ValueError("input_ids required for MDM task")
                
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Token embeddings
            token_embeds = self.token_embedding(input_ids)
            
            # Position embeddings
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeds = self.position_embedding(position_ids)
            
            # Combine embeddings
            hidden_states = token_embeds + position_embeds
            
            # Apply MAMBA layers
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = layer(hidden_states)
            
            # Final layer norm
            hidden_states = self.ln_f(hidden_states)
            
            # MDM head
            outputs["mdm_logits"] = self.mdm_head(hidden_states, masked_positions)
            outputs["hidden_states"] = hidden_states
            
        elif task == "ntp":
            if track_pair is None:
                raise ValueError("track_pair required for NTP task")
                
            track1_ids, track2_ids = track_pair
            
            # Encode both tracks
            track1_embeds = self._encode_single_track(track1_ids)
            track2_embeds = self._encode_single_track(track2_ids)
            
            # NTP head
            outputs["ntp_logits"] = self.ntp_head(track1_embeds, track2_embeds)
        
        return outputs
    
    def _encode_single_track(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a single track to get its representation"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Apply MAMBA layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Average pooling over sequence length to get track representation
        track_embedding = hidden_states.mean(dim=1)
        
        return track_embedding


class MDMHead(nn.Module):
    """Masked Detector Module prediction head"""
    
    def __init__(self, d_model: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            masked_positions: (batch, num_masked) - indices of masked positions
        Returns:
            logits: (batch, num_masked, vocab_size) or (batch, seq_len, vocab_size)
        """
        # Project through dense layer
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # If masked_positions provided, only compute logits for those positions
        if masked_positions is not None:
            batch_size = hidden_states.shape[0]
            masked_hidden = torch.gather(
                hidden_states,
                1,
                masked_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            )
            logits = self.decoder(masked_hidden)
        else:
            logits = self.decoder(hidden_states)
            
        return logits


def create_model(config: Dict, tokenizer: DetectorTokenizer) -> TrackingMAMBA:
    """Factory function to create the model"""
    return TrackingMAMBA(config, tokenizer)


class NTPHead(nn.Module):
    """Next Track Prediction head for momentum comparison"""
    
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
    def forward(self, track1_embeds: torch.Tensor, track2_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compare two tracks to predict which has higher momentum
        
        Args:
            track1_embeds: (batch, d_model)
            track2_embeds: (batch, d_model)
        Returns:
            logits: (batch, 2) - classification logits
        """
        # Concatenate track embeddings
        combined = torch.cat([track1_embeds, track2_embeds], dim=-1)
        logits = self.projection(combined)
        return logits
