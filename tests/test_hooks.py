"""Unit tests for transformer hooks."""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from attnflow.hooks.transformer_hooks import TransformerHookManager
from attnflow.core.memory_stats import MemoryStats


class SimpleAttentionModule(nn.Module):
    """Simple attention module for testing."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleModel(nn.Module):
    """Simple model with attention layers for testing."""
    
    def __init__(self):
        super().__init__()
        self.self_attention = SimpleAttentionModule()
        self.cross_attn = SimpleAttentionModule()
        self.linear = nn.Linear(64, 64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x)
        x = self.cross_attn(x)
        return self.linear(x)


class TestTransformerHookManager:
    """Tests for TransformerHookManager."""
    
    def test_initialization(self) -> None:
        """Test hook manager initialization."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        assert manager.model is model
        assert manager.memory_stats is stats
        assert len(manager._hook_handles) == 0
    
    def test_is_attention_layer(self) -> None:
        """Test attention layer detection."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        # Should detect attention layers
        assert manager._is_attention_layer("self_attention")
        assert manager._is_attention_layer("cross_attn")
        assert manager._is_attention_layer("attention")
        assert manager._is_attention_layer("ATTENTION")  # case-insensitive
        
        # Should not detect non-attention layers
        assert not manager._is_attention_layer("linear")
        assert not manager._is_attention_layer("embeddings")
    
    def test_register_hooks(self) -> None:
        """Test hook registration."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        hook_count = manager.register_hooks()
        
        # Should register hooks on self_attention and cross_attn
        assert hook_count == 2
        assert len(manager._hook_handles) == 2
    
    def test_unregister_hooks(self) -> None:
        """Test hook unregistration."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        manager.register_hooks()
        assert len(manager._hook_handles) == 2
        
        manager.unregister_hooks()
        assert len(manager._hook_handles) == 0
    
    def test_extract_tensor_shape_from_tensor(self) -> None:
        """Test extracting shape from tensor output."""
        shape = TransformerHookManager._extract_tensor_shape(
            torch.randn(2, 10, 64)
        )
        assert shape == (2, 10, 64)
    
    def test_extract_tensor_shape_from_tuple(self) -> None:
        """Test extracting shape from tuple output."""
        tensor = torch.randn(4, 20, 128)
        shape = TransformerHookManager._extract_tensor_shape((tensor,))
        assert shape == (4, 20, 128)
    
    def test_extract_tensor_shape_invalid(self) -> None:
        """Test extracting shape from invalid output."""
        # Should return None for non-tensor outputs
        assert TransformerHookManager._extract_tensor_shape(None) is None
        assert TransformerHookManager._extract_tensor_shape("not a tensor") is None
        assert TransformerHookManager._extract_tensor_shape([]) is None
    
    def test_extract_tensor_shape_low_dim(self) -> None:
        """Test extracting shape from low-dimensional tensor."""
        # Less than 3 dimensions should return None
        shape = TransformerHookManager._extract_tensor_shape(torch.randn(10))
        assert shape is None
    
    def test_estimate_cache_size(self) -> None:
        """Test KV cache size estimation."""
        k_size, v_size = TransformerHookManager._estimate_cache_size(
            batch_size=4,
            seq_len=512,
            hidden_dim=768,
            num_heads=12,
            dtype_bytes=4
        )
        
        # Expected: 4 * 512 * 768 * 4
        expected = 4 * 512 * 768 * 4
        assert k_size == expected
        assert v_size == expected
    
    def test_hook_execution(self) -> None:
        """Test that hooks are executed during forward pass."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        manager.register_hooks()
        
        # Forward pass
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            model(x)
        
        # Check that snapshots were recorded
        all_layers = stats.get_all_layers()
        assert len(all_layers) > 0
        
        # Verify memory was recorded
        for layer in all_layers:
            snapshots = stats.get_snapshots(layer)
            assert len(snapshots) > 0
            assert snapshots[0].sequence_length == 10
        
        manager.unregister_hooks()
    
    def test_multiple_forward_passes(self) -> None:
        """Test tracking across multiple forward passes."""
        model = SimpleModel()
        stats = MemoryStats()
        manager = TransformerHookManager(model, stats)
        
        manager.register_hooks()
        
        # Run with different sequence lengths
        for seq_len in [10, 20, 30]:
            x = torch.randn(2, seq_len, 64)
            with torch.no_grad():
                model(x)
        
        # Check that all snapshots were recorded
        for layer in stats.get_all_layers():
            snapshots = stats.get_snapshots(layer)
            assert len(snapshots) == 3
            
            seq_lens = [s.sequence_length for s in snapshots]
            assert seq_lens == [10, 20, 30]
        
        manager.unregister_hooks()
