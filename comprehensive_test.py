#!/usr/bin/env python
"""Comprehensive functional test script for AttnFlow."""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attnflow import AttentionTracker
from attnflow.core.memory_stats import MemoryStats, LayerMemorySnapshot
from attnflow.viz import print_memory_summary, print_memory_timeline, Visualizer


class SimpleTransformerModel(nn.Module):
    """Simple test model with attention layers."""
    
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


def test_req1_hook_transformer():
    """Requirement 1: Hook transformer forward process."""
    print("\n" + "="*70)
    print("TEST 1: Hook Transformer Forward Process")
    print("="*70)
    
    model = SimpleTransformerModel()
    tracker = AttentionTracker(model, enable_logging=False)
    
    print("✓ Tracker initialized")
    
    # Register hooks
    tracker.register_hooks()
    assert tracker._hooks_registered, "Hooks not registered"
    print(f"✓ Hooks registered: {len(tracker._hook_manager._hook_handles)} hooks")
    
    # Run forward pass
    tracker.start_tracking()
    input_ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        output = model(input_ids)
    
    print("✓ Forward pass executed with hooks active")
    
    # Cleanup
    tracker.stop_tracking()
    tracker.unregister_hooks()
    print("✓ Hooks unregistered")
    
    return True


def test_req2_tensor_capture():
    """Requirement 2: Capture attention tensor sizes."""
    print("\n" + "="*70)
    print("TEST 2: Capture Attention Tensor Sizes")
    print("="*70)
    
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
    
    assert len(layers) > 0, "No layers captured"
    print(f"✓ Captured {len(layers)} attention layers")
    
    # Verify sequence length
    for layer in layers:
        snapshots = stats.get_snapshots(layer)
        if snapshots:
            captured_seq_len = snapshots[-1].sequence_length
            assert captured_seq_len == seq_len, f"Seq len mismatch: {captured_seq_len} vs {seq_len}"
    
    print(f"✓ Tensor shapes correctly extracted (batch={batch_size}, seq_len={seq_len})")
    
    tracker.stop_tracking()
    tracker.unregister_hooks()
    return True


def test_req3_kv_cache_memory():
    """Requirement 3: KV cache memory growth statistics."""
    print("\n" + "="*70)
    print("TEST 3: KV Cache Memory Growth Statistics")
    print("="*70)
    
    model = SimpleTransformerModel(hidden_dim=64)
    tracker = AttentionTracker(model, enable_logging=False)
    tracker.register_hooks()
    tracker.start_tracking()
    
    memory_snapshots = []
    
    with torch.no_grad():
        for seq_len in [5, 10, 20]:
            input_ids = torch.randint(0, 100, (2, seq_len))
            output = model(input_ids)
            memory_snapshots.append((seq_len, tracker.get_memory_stats()))
    
    stats = tracker.get_memory_stats()
    summary = stats.get_summary()
    
    assert len(summary) > 0, "No memory stats recorded"
    print(f"✓ Memory tracked for {len(summary)} layers")
    
    # Verify memory values
    for layer_name, layer_stats in summary.items():
        peak_mb = layer_stats["peak_memory_mb"]
        peak_seq = layer_stats["peak_sequence_length"]
        assert peak_mb >= 0, "Invalid peak memory"
        assert peak_seq > 0, "Invalid sequence length"
        print(f"  Layer {layer_name}: {peak_mb:.2f} MB (max_seq={peak_seq})")
    
    tracker.stop_tracking()
    tracker.unregister_hooks()
    return True


def test_req4_visualization():
    """Requirement 4: Simple visualization (CLI + matplotlib)."""
    print("\n" + "="*70)
    print("TEST 4: Visualization Output (CLI & Matplotlib)")
    print("="*70)
    
    model = SimpleTransformerModel()
    tracker = AttentionTracker(model, enable_logging=False)
    tracker.register_hooks()
    tracker.start_tracking()
    
    with torch.no_grad():
        for _ in range(3):
            input_ids = torch.randint(0, 100, (2, 10))
            output = model(input_ids)
    
    stats = tracker.get_memory_stats()
    
    # Test CLI output
    print("\n--- CLI Memory Summary ---")
    print_memory_summary(stats)
    print("✓ CLI summary output working")
    
    # Test CLI timeline
    layers = stats.get_all_layers()
    if layers:
        print(f"\n--- CLI Timeline for {layers[0]} ---")
        print_memory_timeline(stats, layers[0])
        print("✓ CLI timeline output working")
    
    # Test matplotlib plots
    print("\n--- Matplotlib Visualization ---")
    viz = Visualizer()
    
    fig1 = viz.plot_peak_memory_comparison(stats)
    assert fig1 is not None, "Peak memory plot failed"
    print("✓ Peak memory comparison plot generated")
    
    fig2 = viz.plot_memory_timeline(stats)
    assert fig2 is not None, "Memory timeline plot failed"
    print("✓ Memory timeline plot generated")
    
    tracker.stop_tracking()
    tracker.unregister_hooks()
    return True


def test_req5_memory_timeline():
    """Requirement 5: Display each layer's memory timeline."""
    print("\n" + "="*70)
    print("TEST 5: Memory Usage Timeline per Layer")
    print("="*70)
    
    model = SimpleTransformerModel()
    tracker = AttentionTracker(model, enable_logging=False)
    tracker.register_hooks()
    tracker.start_tracking()
    
    with torch.no_grad():
        for i in range(3):
            input_ids = torch.randint(0, 100, (2, 10 + i*5))
            output = model(input_ids)
            time.sleep(0.01)  # Small delay for timestamp separation
    
    stats = tracker.get_memory_stats()
    
    for layer in stats.get_all_layers():
        timestamps, memory_values = stats.get_memory_timeline(layer)
        assert len(timestamps) > 0, f"No timeline for {layer}"
        assert len(timestamps) == len(memory_values), "Timeline length mismatch"
        
        print(f"✓ {layer}: {len(timestamps)} snapshots")
        # Print first few values
        for t, mem in zip(timestamps[:3], memory_values[:3]):
            print(f"    t={t:.4f}s, mem={mem:.2f}MB")
    
    tracker.stop_tracking()
    tracker.unregister_hooks()
    return True


def test_context_manager():
    """Test: Context manager convenience API."""
    print("\n" + "="*70)
    print("TEST 6: Context Manager API")
    print("="*70)
    
    model = SimpleTransformerModel()
    
    with AttentionTracker(model, enable_logging=False) as tracker:
        assert tracker._hooks_registered, "Hooks not registered in context"
        assert tracker._is_tracking, "Not tracking in context"
        
        input_ids = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            output = model(input_ids)
        
        stats = tracker.get_memory_stats()
        assert len(stats.get_all_layers()) > 0, "No data collected"
    
    assert not tracker._hooks_registered, "Hooks not cleaned up after context"
    assert not tracker._is_tracking, "Still tracking after context"
    
    print("✓ Context manager automatically handles hook registration/cleanup")
    print("✓ Tracking state correctly managed")
    return True


def test_data_structures():
    """Test: Core data structures."""
    print("\n" + "="*70)
    print("TEST 7: Core Data Structures")
    print("="*70)
    
    # Test LayerMemorySnapshot
    snapshot = LayerMemorySnapshot(
        layer_name="test",
        timestamp=0.5,
        k_cache_size=1024,
        v_cache_size=1024,
        sequence_length=10
    )
    assert snapshot.total_memory == 2048, "Total memory calculation wrong"
    assert snapshot.memory_mb > 0, "Memory MB calculation wrong"
    print("✓ LayerMemorySnapshot dataclass working")
    
    # Test MemoryStats
    stats = MemoryStats()
    stats.record_snapshot("layer1", 1000, 1000, 10)
    stats.record_snapshot("layer1", 2000, 2000, 20)
    
    assert len(stats.get_all_layers()) == 1, "Layer not recorded"
    assert len(stats.get_snapshots("layer1")) == 2, "Snapshots not recorded"
    assert stats.get_peak_memory("layer1") == 4000, "Peak memory wrong"
    
    summary = stats.get_summary()
    assert "layer1" in summary, "Summary missing layer"
    assert summary["layer1"]["snapshot_count"] == 2, "Snapshot count wrong"
    
    print("✓ MemoryStats tracking and queries working")
    return True


def run_all_tests():
    """Run all requirement tests."""
    print("\n" + "="*70)
    print("AttnFlow Comprehensive Requirement Test Suite")
    print("="*70)
    
    tests = [
        ("1. Hook Transformer", test_req1_hook_transformer),
        ("2. Tensor Capture", test_req2_tensor_capture),
        ("3. KV Cache Memory", test_req3_kv_cache_memory),
        ("4. Visualization", test_req4_visualization),
        ("5. Memory Timeline", test_req5_memory_timeline),
        ("6. Context Manager", test_context_manager),
        ("7. Data Structures", test_data_structures),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✓ PASS", None))
        except Exception as e:
            results.append((test_name, "✗ FAIL", str(e)))
            print(f"\n✗ ERROR in {test_name}: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status, _ in results if "PASS" in status)
    total = len(results)
    
    for test_name, status, error in results:
        print(f"{status}: {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
