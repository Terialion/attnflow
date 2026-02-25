"""Integration tests for AttnFlow.

Tests the complete workflow of attention tracking end-to-end.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from attnflow import AttentionTracker
from attnflow.viz import Visualizer
from demo.attention_model import SimpleTransformerModel


class TestIntegration:
    """Integration tests for complete AttnFlow workflow."""
    
    def test_complete_workflow(self) -> None:
        """Test complete tracking workflow."""
        model = SimpleTransformerModel(num_layers=2)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            # Run forward passes with different sequence lengths
            for seq_len in [10, 20, 30]:
                input_ids = torch.randint(0, 1000, (4, seq_len))
                with torch.no_grad():
                    model(input_ids)
            
            # Get statistics
            stats = tracker.get_memory_stats()
            summary = tracker.get_summary()
            
            # Verify results
            assert len(stats.get_all_layers()) > 0
            assert len(summary) > 0
            
            for layer_name, layer_stats in summary.items():
                assert "peak_memory_mb" in layer_stats
                assert "peak_sequence_length" in layer_stats
                assert "snapshot_count" in layer_stats
                assert layer_stats["snapshot_count"] == 3
    
    def test_context_manager_cleanup(self) -> None:
        """Test that context manager properly cleans up hooks."""
        model = SimpleTransformerModel(num_layers=2)
        tracker = None
        
        with AttentionTracker(model, enable_logging=False) as t:
            tracker = t
            assert tracker._hooks_registered
            assert tracker._is_tracking
        
        # After context exit
        assert not tracker._hooks_registered
        assert not tracker._is_tracking
    
    def test_reset_between_batches(self) -> None:
        """Test resetting statistics between batches."""
        model = SimpleTransformerModel(num_layers=2)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            # First batch
            for seq_len in [10, 20]:
                x = torch.randint(0, 1000, (4, seq_len))
                with torch.no_grad():
                    model(x)
            
            stats1 = tracker.get_memory_stats()
            count1 = len(stats1.get_all_layers())
            
            # Reset
            tracker.reset_stats()
            
            # Second batch
            x = torch.randint(0, 1000, (4, 30))
            with torch.no_grad():
                model(x)
            
            stats2 = tracker.get_memory_stats()
            count2 = len(stats2.get_all_layers())
            
            # Should have same layers but different statistics
            assert count1 == count2
            
            for layer in stats1.get_all_layers():
                snapshots2 = stats2.get_snapshots(layer)
                assert len(snapshots2) == 1  # Only one snapshot from second batch
    
    def test_visualization_generation(self) -> None:
        """Test visualization generation."""
        model = SimpleTransformerModel(num_layers=2)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            for seq_len in [10, 20, 30]:
                x = torch.randint(0, 1000, (4, seq_len))
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            viz = Visualizer()
            
            # Test timeline plot generation
            fig = viz.plot_memory_timeline(stats)
            assert fig is not None
            
            # Test comparison plot generation
            fig = viz.plot_peak_memory_comparison(stats)
            assert fig is not None
    
    def test_multiple_models(self) -> None:
        """Test tracking multiple different models."""
        model1 = SimpleTransformerModel(num_layers=2)
        model2 = SimpleTransformerModel(num_layers=3)
        
        # Track first model
        with AttentionTracker(model1, enable_logging=False) as tracker1:
            x = torch.randint(0, 1000, (4, 20))
            with torch.no_grad():
                model1(x)
            
            stats1 = tracker1.get_memory_stats()
            layers1 = stats1.get_all_layers()
        
        # Track second model
        with AttentionTracker(model2, enable_logging=False) as tracker2:
            x = torch.randint(0, 1000, (4, 20))
            with torch.no_grad():
                model2(x)
            
            stats2 = tracker2.get_memory_stats()
            layers2 = stats2.get_all_layers()
        
        # Both should have tracked layers
        assert len(layers1) > 0
        assert len(layers2) > 0
    
    def test_large_model(self) -> None:
        """Test tracking a larger model."""
        model = SimpleTransformerModel(num_layers=10)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            for seq_len in [64, 128, 256]:
                x = torch.randint(0, 1000, (2, seq_len))
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            summary = tracker.get_summary()
            
            # All layers should be tracked
            all_layers = stats.get_all_layers()
            assert len(all_layers) > 0
            
            # Each layer should have multiple snapshots
            for layer in all_layers:
                snapshots = stats.get_snapshots(layer)
                assert len(snapshots) == 3
    
    def test_batch_size_variation(self) -> None:
        """Test tracking with different batch sizes."""
        model = SimpleTransformerModel(num_layers=2)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            batch_sizes = [1, 4, 8]
            for batch_size in batch_sizes:
                x = torch.randint(0, 1000, (batch_size, 100))
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            
            # Should have tracked memory for different batch sizes
            for layer in stats.get_all_layers():
                snapshots = stats.get_snapshots(layer)
                assert len(snapshots) == 3
    
    @pytest.mark.slow
    def test_extended_sequence_tracking(self) -> None:
        """Test tracking very long sequences (may be slow)."""
        model = SimpleTransformerModel(num_layers=2)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            for seq_len in [100, 200, 512]:
                x = torch.randint(0, 1000, (2, seq_len))
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            summary = tracker.get_summary()
            
            assert len(summary) > 0
            
            # Peak sequence length should be 512
            for layer_stats in summary.values():
                assert layer_stats["peak_sequence_length"] == 512
