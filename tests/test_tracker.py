"""Unit tests for attention tracker."""

import pytest
import torch
import torch.nn as nn

from attnflow.core.tracker import AttentionTracker


class DummyModel(nn.Module):
    """Dummy model with self attention for testing."""
    
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x)
        return self.fc(x)


class TestAttentionTracker:
    """Tests for AttentionTracker."""
    
    def test_initialization(self) -> None:
        """Test tracker initialization."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        assert tracker.model is model
        assert not tracker.enable_logging
        assert not tracker._hooks_registered
        assert not tracker._is_tracking
    
    def test_register_hooks(self) -> None:
        """Test hook registration."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        
        assert tracker._hooks_registered
    
    def test_register_hooks_idempotent(self) -> None:
        """Test that registering hooks twice is safe."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        tracker.register_hooks()
        
        # Should only be registered once, second call should skip
        assert tracker._hooks_registered
    
    def test_unregister_hooks(self) -> None:
        """Test hook unregistration."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        assert tracker._hooks_registered
        
        tracker.unregister_hooks()
        assert not tracker._hooks_registered
    
    def test_start_tracking_requires_hooks(self) -> None:
        """Test that tracking requires hooks to be registered."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        with pytest.raises(RuntimeError):
            tracker.start_tracking()
    
    def test_start_and_stop_tracking(self) -> None:
        """Test starting and stopping tracking."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        
        assert not tracker.is_tracking()
        
        tracker.start_tracking()
        assert tracker.is_tracking()
        
        tracker.stop_tracking()
        assert not tracker.is_tracking()
    
    def test_reset_stats(self) -> None:
        """Test resetting statistics."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        tracker.start_tracking()
        
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            model(x)
        
        stats = tracker.get_memory_stats()
        assert len(stats.get_all_layers()) > 0
        
        tracker.reset_stats()
        
        stats = tracker.get_memory_stats()
        assert len(stats.get_all_layers()) == 0
    
    def test_get_memory_stats(self) -> None:
        """Test getting memory statistics."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        tracker.start_tracking()
        
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            model(x)
        
        stats = tracker.get_memory_stats()
        
        assert stats is not None
        assert len(stats.get_all_layers()) > 0
    
    def test_get_summary(self) -> None:
        """Test getting summary."""
        model = DummyModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        tracker.register_hooks()
        tracker.start_tracking()
        
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            model(x)
        
        summary = tracker.get_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) > 0
    
    def test_context_manager(self) -> None:
        """Test using tracker as context manager."""
        model = DummyModel()
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            assert tracker._hooks_registered
            assert tracker._is_tracking
            
            x = torch.randn(2, 10, 64)
            with torch.no_grad():
                model(x)
            
            stats = tracker.get_memory_stats()
            assert len(stats.get_all_layers()) > 0
        
        # After exiting context, hooks should be unregistered
        assert not tracker._hooks_registered
        assert not tracker._is_tracking
    
    def test_multiple_forward_passes(self) -> None:
        """Test tracking across multiple forward passes."""
        model = DummyModel()
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            for seq_len in [10, 20, 30]:
                x = torch.randn(2, seq_len, 64)
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            
            for layer in stats.get_all_layers():
                snapshots = stats.get_snapshots(layer)
                seq_lengths = [s.sequence_length for s in snapshots]
                # Should have 3 snapshots from 3 forward passes
                assert len(snapshots) == 3
