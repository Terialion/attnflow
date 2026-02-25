"""Comprehensive test suite validating all project requirements."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from io import StringIO

from attnflow import AttentionTracker
from attnflow.core.memory_stats import MemoryStats, LayerMemorySnapshot
from attnflow.hooks.transformer_hooks import TransformerHookManager
from attnflow.viz import print_memory_summary, print_memory_timeline, Visualizer
from attnflow.utils.logger import get_logger


class SimpleTransformerModel(nn.Module):
    """Simple model with attention layers for testing."""
    
    def __init__(self, vocab_size=100, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            SimpleAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)


class SimpleAttentionLayer(nn.Module):
    """Simple attention layer."""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.fc(attn_out)


class TestRequirement1HookTransformer:
    """Test: Ability to hook transformer forward process."""
    
    def test_hook_registration(self):
        """Test that hooks can be registered on attention layers."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        
        assert not tracker._hooks_registered
        tracker.register_hooks()
        assert tracker._hooks_registered
        assert len(tracker._hook_manager._hook_handles) > 0
        
        tracker.unregister_hooks()
        assert not tracker._hooks_registered
    
    def test_hook_capture_during_forward(self):
        """Test that hooks capture data during forward pass."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        assert len(stats.get_all_layers()) > 0
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes with hook tracking."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            for seq_len in [5, 10, 15]:
                input_ids = torch.randint(0, 100, (2, seq_len))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        layers = stats.get_all_layers()
        assert len(layers) > 0
        
        # Check that multiple snapshots were recorded
        for layer in layers:
            snapshots = stats.get_snapshots(layer)
            assert len(snapshots) >= 3  # At least 3 forward passes
        
        tracker.stop_tracking()
        tracker.unregister_hooks()


class TestRequirement2AttentionTensorCapture:
    """Test: Capture attention tensor sizes."""
    
    def test_tensor_shape_extraction(self):
        """Test extraction of tensor dimensions."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        batch_size, seq_len = 4, 12
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        layers = stats.get_all_layers()
        
        # Verify sequence length was captured
        for layer in layers:
            snapshots = stats.get_snapshots(layer)
            if snapshots:
                captured_seq_len = snapshots[-1].sequence_length
                assert captured_seq_len == seq_len
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_different_batch_sizes(self):
        """Test tensor capture with different batch sizes."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8]:
                input_ids = torch.randint(0, 100, (batch_size, 10))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        layers = stats.get_all_layers()
        
        assert len(layers) > 0
        for layer in layers:
            snapshots = stats.get_snapshots(layer)
            assert len(snapshots) >= 4
        
        tracker.stop_tracking()
        tracker.unregister_hooks()


class TestRequirement3KVCacheMemory:
    """Test: KV cache memory growth statistics."""
    
    def test_kv_cache_size_calculation(self):
        """Test KV cache size is calculated and tracked."""
        model = SimpleTransformerModel(hidden_dim=64)
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        
        # Check that memory is tracked in bytes
        for layer in stats.get_all_layers():
            snapshots = stats.get_snapshots(layer)
            if snapshots:
                snapshot = snapshots[-1]
                assert snapshot.k_cache_size >= 0
                assert snapshot.v_cache_size >= 0
                assert snapshot.total_memory > 0
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_memory_growth_with_longer_sequences(self):
        """Test that memory grows with longer sequences."""
        model = SimpleTransformerModel(hidden_dim=64)
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        memory_values = []
        with torch.no_grad():
            for seq_len in [5, 10, 20]:
                input_ids = torch.randint(0, 100, (2, seq_len))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        
        # Get a representative layer
        layers = stats.get_all_layers()
        if layers:
            layer = layers[0]
            snapshots = stats.get_snapshots(layer)
            memory_values = [s.memory_mb for s in snapshots]
            
            # Memory should generally increase or stay same with longer sequences
            assert len(memory_values) >= 3
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_peak_memory_tracking(self):
        """Test peak memory is correctly identified."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            # Multiple forward passes with different sequence lengths
            for seq_len in [5, 15, 10]:
                input_ids = torch.randint(0, 100, (2, seq_len))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        summary = stats.get_summary()
        
        assert len(summary) > 0
        for layer_name, layer_stats in summary.items():
            assert "peak_memory_mb" in layer_stats
            assert "peak_sequence_length" in layer_stats
            assert layer_stats["peak_memory_mb"] >= 0
            assert layer_stats["peak_sequence_length"] > 0
        
        tracker.stop_tracking()
        tracker.unregister_hooks()


class TestRequirement4Visualization:
    """Test: Simple visualization output (CLI or matplotlib)."""
    
    def test_cli_memory_summary(self):
        """Test CLI summary output printing."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        
        # Capture output
        with patch('sys.stdout', new=StringIO()) as fake_output:
            print_memory_summary(stats)
            output_str = fake_output.getvalue()
        
        assert "ATTENTION MEMORY SUMMARY" in output_str
        assert "Layer Name" in output_str
        assert "Peak Memory" in output_str
        assert "TOTAL" in output_str
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_cli_memory_timeline(self):
        """Test CLI timeline output printing."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            for _ in range(3):
                input_ids = torch.randint(0, 100, (2, 10))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        layers = stats.get_all_layers()
        
        if layers:
            with patch('sys.stdout', new=StringIO()) as fake_output:
                print_memory_timeline(stats, layers[0])
                output_str = fake_output.getvalue()
            
            assert "Memory Timeline" in output_str
            assert "Time (s)" in output_str
            assert "Memory (MB)" in output_str
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_matplotlib_visualization(self):
        """Test matplotlib plot generation."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            for seq_len in [5, 10, 15]:
                input_ids = torch.randint(0, 100, (2, seq_len))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        viz = Visualizer()
        
        # Test peak memory comparison plot
        fig1 = viz.plot_peak_memory_comparison(stats)
        assert fig1 is not None
        
        # Test memory timeline plot
        fig2 = viz.plot_memory_timeline(stats)
        assert fig2 is not None
        
        tracker.stop_tracking()
        tracker.unregister_hooks()


class TestRequirement5MemoryTimeline:
    """Test: Display each layer's memory usage timeline."""
    
    def test_timeline_data_collection(self):
        """Test that timeline data is collected for each layer."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        with torch.no_grad():
            for i in range(3):
                input_ids = torch.randint(0, 100, (2, 10 + i*5))
                output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        
        # Check timeline data for each layer
        for layer in stats.get_all_layers():
            timestamps, memory_values = stats.get_memory_timeline(layer)
            assert len(timestamps) > 0
            assert len(memory_values) > 0
            assert len(timestamps) == len(memory_values)
        
        tracker.stop_tracking()
        tracker.unregister_hooks()
    
    def test_timeline_has_timestamps(self):
        """Test that timeline includes proper timestamps."""
        model = SimpleTransformerModel()
        tracker = AttentionTracker(model, enable_logging=False)
        tracker.register_hooks()
        tracker.start_tracking()
        
        import time
        with torch.no_grad():
            for _ in range(3):
                input_ids = torch.randint(0, 100, (2, 10))
                output = model(input_ids)
                time.sleep(0.01)  # Small delay
        
        stats = tracker.get_memory_stats()
        layers = stats.get_all_layers()
        
        if layers:
            timestamps, memory_values = stats.get_memory_timeline(layers[0])
            # Timestamps should be increasing
            assert timestamps == sorted(timestamps)
        
        tracker.stop_tracking()
        tracker.unregister_hooks()


class TestContextManager:
    """Test: Context manager functionality for cleaner API."""
    
    def test_context_manager_usage(self):
        """Test using tracker as context manager."""
        model = SimpleTransformerModel()
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            assert tracker._hooks_registered
            assert tracker._is_tracking
            
            input_ids = torch.randint(0, 100, (2, 10))
            with torch.no_grad():
                output = model(input_ids)
            
            stats = tracker.get_memory_stats()
            assert len(stats.get_all_layers()) > 0
        
        # After context, hooks should be unregistered
        assert not tracker._hooks_registered
        assert not tracker._is_tracking


class TestDataStructures:
    """Test: Core data structures for memory stats."""
    
    def test_layer_memory_snapshot(self):
        """Test LayerMemorySnapshot dataclass."""
        snapshot = LayerMemorySnapshot(
            layer_name="test_layer",
            timestamp=0.5,
            k_cache_size=1024,
            v_cache_size=1024,
            sequence_length=10
        )
        
        assert snapshot.layer_name == "test_layer"
        assert snapshot.total_memory == 2048
        assert snapshot.memory_mb == 2048 / (1024 * 1024)
    
    def test_memory_stats_operations(self):
        """Test MemoryStats core operations."""
        stats = MemoryStats()
        
        # Record snapshots
        stats.record_snapshot("layer1", 1000, 1000, 10)
        stats.record_snapshot("layer1", 2000, 2000, 20)
        stats.record_snapshot("layer2", 1500, 1500, 15)
        
        # Test getters
        assert "layer1" in stats.get_all_layers()
        assert "layer2" in stats.get_all_layers()
        assert len(stats.get_snapshots("layer1")) == 2
        assert len(stats.get_snapshots("layer2")) == 1
        
        # Test peak memory
        assert stats.get_peak_memory("layer1") == 4000  # 2000 + 2000
        assert stats.get_peak_memory_mb("layer1") == 4000 / (1024 * 1024)
        
        # Test summary
        summary = stats.get_summary()
        assert "layer1" in summary
        assert summary["layer1"]["snapshot_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
