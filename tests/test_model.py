"""
Test suite for MAMBA-TrackingBERT
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.tokenizer import DetectorTokenizer
from src.models.mamba_block import MambaBlock, ResidualMambaBlock
from src.models.tracking_mamba import TrackingMAMBA, MDMHead, NTPHead


def test_tokenizer():
    """Test detector tokenizer functionality"""
    tokenizer = DetectorTokenizer(vocab_size=1000)
    
    # Test vocabulary building
    modules = [(7, 2, 100), (8, 4, 200), (9, 6, 300)]
    tokenizer.build_vocab(modules)
    
    assert tokenizer.is_built
    assert len(tokenizer.token_to_id) > len(tokenizer.special_tokens)
    
    # Test encoding
    track_modules = [(7, 2, 100), (8, 4, 200)]
    encoded = tokenizer.encode_track(track_modules, max_length=10)
    
    assert 'input_ids' in encoded
    assert 'attention_mask' in encoded
    assert len(encoded['input_ids']) == 10
    assert len(encoded['attention_mask']) == 10
    
    # Test masking
    masked_ids, positions = tokenizer.create_masked_input(
        encoded['input_ids'],
        mask_rate=0.5
    )
    
    assert len(masked_ids) == len(encoded['input_ids'])
    assert len(positions) > 0
    assert masked_ids[positions[0]] == tokenizer.mask_token_id


def test_mamba_block():
    """Test MAMBA block"""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    block = MambaBlock(d_model=d_model)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    
    # Test residual block
    res_block = ResidualMambaBlock(d_model=d_model)
    output = res_block(x)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_tracking_mamba():
    """Test main model"""
    config = {
        'model': {
            'd_model': 64,
            'n_layers': 2,
            'vocab_size': 1000,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.1,
            'mdm_hidden_dim': 128,
            'ntp_hidden_dim': 64,
            'gradient_checkpointing': False,
        }
    }
    
    tokenizer = DetectorTokenizer(vocab_size=config['model']['vocab_size'])
    modules = [(7, i, j) for i in range(10) for j in range(100)]
    tokenizer.build_vocab(modules)
    
    model = TrackingMAMBA(config, tokenizer)
    
    batch_size = 2
    seq_len = 10
    
    # Test MDM forward pass
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    masked_positions = torch.tensor([[2, 5], [1, 7]])
    
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        masked_positions=masked_positions,
        task="mdm"
    )
    
    assert 'mdm_logits' in outputs
    assert outputs['mdm_logits'].shape == (batch_size, 2, config['model']['vocab_size'])
    
    # Test NTP forward pass
    track1 = torch.randint(0, 100, (batch_size, seq_len))
    track2 = torch.randint(0, 100, (batch_size, seq_len))
    
    outputs = model(
        input_ids=None,
        track_pair=(track1, track2),
        task="ntp"
    )
    
    assert 'ntp_logits' in outputs
    assert outputs['ntp_logits'].shape == (batch_size, 2)


def test_mdm_head():
    """Test MDM head"""
    d_model = 64
    hidden_dim = 128
    vocab_size = 1000
    
    head = MDMHead(d_model, hidden_dim, vocab_size)
    
    batch_size = 2
    seq_len = 10
    num_masked = 3
    
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    masked_positions = torch.randint(0, seq_len, (batch_size, num_masked))
    
    logits = head(hidden_states, masked_positions)
    
    assert logits.shape == (batch_size, num_masked, vocab_size)


def test_ntp_head():
    """Test NTP head"""
    d_model = 64
    hidden_dim = 128
    
    head = NTPHead(d_model, hidden_dim)
    
    batch_size = 2
    
    track1_embeds = torch.randn(batch_size, d_model)
    track2_embeds = torch.randn(batch_size, d_model)
    
    logits = head(track1_embeds, track2_embeds)
    
    assert logits.shape == (batch_size, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
