"""Simple demonstration of AttnFlow tracking."""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attnflow import AttentionTracker
from attnflow.viz import print_memory_summary, print_memory_timeline, Visualizer
from demo.attention_model import SimpleTransformerModel


def run_demo():
    """
    Run complete demo of attention tracking.
    
    Demonstrates:
    1. Model initialization
    2. Hook registration
    3. Tracking during forward pass
    4. Memory statistics visualization
    """
    print("="*70)
    print("AttnFlow Demo: Transformer Attention Memory Tracking")
    print("="*70)
    
    # Step 1: Create model
    print("\n[Step 1] Initializing model...")
    model = SimpleTransformerModel(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 2: Create tracker and register hooks
    print("\n[Step 2] Setting up attention tracker...")
    tracker = AttentionTracker(model, enable_logging=True)
    tracker.register_hooks()
    
    # Step 3: Run forward passes with tracking
    print("\n[Step 3] Running forward passes...")
    model.eval()
    tracker.start_tracking()
    
    with torch.no_grad():
        # Run multiple sequences of increasing length
        for seq_len in [10, 20, 30]:
            batch_size = 4
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            print(f"  Processing batch: seq_len={seq_len}, batch_size={batch_size}...", end="")
            output = model(input_ids)
            print(f" ✓")
    
    tracker.stop_tracking()
    
    # Step 4: Visualize results
    print("\n[Step 4] Memory Statistics")
    print("-"*70)
    
    stats = tracker.get_memory_stats()
    print_memory_summary(stats)
    
    # Step 5: Show timeline for each layer
    print("[Step 5] Memory Timeline")
    print("-"*70)
    for layer_name in stats.get_all_layers()[:2]:  # Show first 2 layers
        print_memory_timeline(stats, layer_name)
    
    # Step 6: Generate plots
    print("[Step 6] Generating visualizations...")
    viz = Visualizer()
    
    # Plot 1: Peak memory comparison
    fig1 = viz.plot_peak_memory_comparison(stats)
    fig1.savefig("/tmp/attnflow_peak_memory.png")
    print("  ✓ Saved peak memory plot to /tmp/attnflow_peak_memory.png")
    
    # Plot 2: Memory timeline
    fig2 = viz.plot_memory_timeline(stats)
    fig2.savefig("/tmp/attnflow_timeline.png")
    print("  ✓ Saved timeline plot to /tmp/attnflow_timeline.png")
    
    # Cleanup
    tracker.unregister_hooks()
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


def run_context_manager_demo():
    """
    Demonstrate using AttentionTracker as context manager.
    
    This is a more concise way to use the tracker.
    """
    print("\n" + "="*70)
    print("Context Manager Demo")
    print("="*70)
    
    model = SimpleTransformerModel(vocab_size=500, hidden_dim=128, num_layers=1)
    model.eval()
    
    # Using tracker as context manager
    with AttentionTracker(model) as tracker:
        with torch.no_grad():
            input_ids = torch.randint(0, 500, (2, 15))
            output = model(input_ids)
        
        # Get stats while tracking is still active
        stats = tracker.get_memory_stats()
        print("\nMemory Summary:")
        print_memory_summary(stats)


if __name__ == "__main__":
    # Run main demo
    run_demo()
    
    # Run context manager demo
    run_context_manager_demo()
    
    print("\n✓ All demos completed!")
