"""Performance benchmarks for AttnFlow.

Run with: python benchmarks/benchmark_performance.py

Measures:
- Hook registration time
- Memory tracking overhead
- Summary computation time
- Visualization generation time
"""

import time
import torch
import torch.nn as nn
from typing import List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attnflow import AttentionTracker
from attnflow.viz import Visualizer
from demo.attention_model import SimpleTransformerModel


class BenchmarkRunner:
    """Runs performance benchmarks for AttnFlow."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.results = {}
    
    def benchmark_hook_registration(self, num_layers: int = 10) -> float:
        """
        Benchmark hook registration time.
        
        Args:
            num_layers: Number of transformer layers in test model
            
        Returns:
            Registration time in milliseconds
        """
        model = SimpleTransformerModel(num_layers=num_layers)
        tracker = AttentionTracker(model, enable_logging=False)
        
        start = time.time()
        tracker.register_hooks()
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        tracker.unregister_hooks()
        return elapsed
    
    def benchmark_forward_pass_overhead(
        self,
        num_layers: int = 2,
        seq_len: int = 256,
        num_passes: int = 10
    ) -> Tuple[float, float]:
        """
        Benchmark overhead of tracking vs no tracking.
        
        Args:
            num_layers: Number of transformer layers
            seq_len: Sequence length
            num_passes: Number of forward passes
            
        Returns:
            Tuple of (time_with_tracking_ms, time_without_tracking_ms)
        """
        model = SimpleTransformerModel(num_layers=num_layers)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (4, seq_len))
        
        # Measure without tracking
        with torch.no_grad():
            start = time.time()
            for _ in range(num_passes):
                model(input_ids)
            time_without = (time.time() - start) * 1000
        
        # Measure with tracking
        with AttentionTracker(model, enable_logging=False) as tracker:
            with torch.no_grad():
                start = time.time()
                for _ in range(num_passes):
                    model(input_ids)
                time_with = (time.time() - start) * 1000
        
        return time_with, time_without
    
    def benchmark_memory_stats_query(
        self,
        num_layers: int = 10,
        num_snapshots: int = 100
    ) -> Tuple[float, float, float]:
        """
        Benchmark memory statistics query performance.
        
        Args:
            num_layers: Number of layers to track
            num_snapshots: Number of snapshots per layer
            
        Returns:
            Tuple of (get_summary_ms, get_peak_ms, get_timeline_ms)
        """
        model = SimpleTransformerModel(num_layers=num_layers)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            # Generate data
            for _ in range(num_snapshots):
                x = torch.randn(4, 10 + _ % 100, 256)
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            layers = stats.get_all_layers()
            
            # Benchmark get_summary
            start = time.time()
            for _ in range(1000):
                stats.get_summary()
            get_summary_time = (time.time() - start) * 1000
            
            # Benchmark get_peak_memory
            start = time.time()
            for _ in range(1000):
                for layer in layers:
                    stats.get_peak_memory(layer)
            get_peak_time = (time.time() - start) * 1000
            
            # Benchmark get_memory_timeline
            start = time.time()
            for _ in range(100):
                for layer in layers:
                    stats.get_memory_timeline(layer)
            get_timeline_time = (time.time() - start) * 1000
        
        return get_summary_time, get_peak_time, get_timeline_time
    
    def benchmark_visualization(
        self,
        num_layers: int = 10,
        num_snapshots: int = 50
    ) -> Tuple[float, float]:
        """
        Benchmark visualization generation time.
        
        Args:
            num_layers: Number of layers
            num_snapshots: Snapshots per layer
            
        Returns:
            Tuple of (timeline_plot_ms, comparison_plot_ms)
        """
        model = SimpleTransformerModel(num_layers=num_layers)
        
        with AttentionTracker(model, enable_logging=False) as tracker:
            # Generate data
            for _ in range(num_snapshots):
                x = torch.randn(4, 10 + _ % 50, 256)
                with torch.no_grad():
                    model(x)
            
            stats = tracker.get_memory_stats()
            viz = Visualizer()
            
            # Benchmark timeline plot
            start = time.time()
            fig = viz.plot_memory_timeline(stats)
            timeline_time = (time.time() - start) * 1000
            
            # Benchmark comparison plot
            start = time.time()
            fig = viz.plot_peak_memory_comparison(stats)
            comparison_time = (time.time() - start) * 1000
        
        return timeline_time, comparison_time
    
    def run_all_benchmarks(self) -> None:
        """Run all benchmarks and print results."""
        print("\n" + "="*70)
        print("AttnFlow Performance Benchmarks".center(70))
        print("="*70 + "\n")
        
        # Hook registration
        print("[1/4] Benchmarking hook registration...")
        registration_time = self.benchmark_hook_registration()
        print(f"  ✓ Hook registration: {registration_time:.2f} ms")
        
        # Forward pass overhead
        print("\n[2/4] Benchmarking forward pass overhead...")
        time_with, time_without = self.benchmark_forward_pass_overhead()
        overhead_pct = ((time_with - time_without) / time_without) * 100
        print(f"  Without tracking: {time_without:.2f} ms")
        print(f"  With tracking:    {time_with:.2f} ms")
        print(f"  Overhead:         {overhead_pct:.1f}%")
        
        # Statistics query
        print("\n[3/4] Benchmarking statistics queries...")
        summary_time, peak_time, timeline_time = self.benchmark_memory_stats_query()
        print(f"  ✓ get_summary (1000x):     {summary_time:.2f} ms")
        print(f"  ✓ get_peak_memory (1000x): {peak_time:.2f} ms")
        print(f"  ✓ get_timeline (100x):     {timeline_time:.2f} ms")
        
        # Visualization
        print("\n[4/4] Benchmarking visualization...")
        timeline_plot, comparison_plot = self.benchmark_visualization()
        print(f"  ✓ Timeline plot: {timeline_plot:.2f} ms")
        print(f"  ✓ Comparison plot: {comparison_plot:.2f} ms")
        
        print("\n" + "="*70)
        print("Benchmark Complete".center(70))
        print("="*70 + "\n")


def main() -> None:
    """Run performance benchmarks."""
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
