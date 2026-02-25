"""Minimal realtime demo for attention memory flow visualization."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

# Add parent directory to path for local execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from attnflow import AttentionTracker
from attnflow.viz import RealtimeMemoryDashboard, print_memory_summary
from demo.attention_model import SimpleTransformerModel


def _build_input(batch_size: int, seq_len: int, vocab_size: int) -> Optional[torch.Tensor]:
    """Build one demo input tensor with boundary checks.

    Args:
        batch_size: Number of samples in the batch.
        seq_len: Token sequence length.
        vocab_size: Vocabulary size upper bound.

    Returns:
        Tensor of token IDs, or ``None`` if parameters are invalid.
    """
    if batch_size <= 0 or seq_len <= 0 or vocab_size <= 1:
        return None
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def run_realtime_demo(total_steps: int = 20, step_delay_s: float = 0.1) -> None:
    """Run a minimal realtime dashboard demo on CPU.

    Args:
        total_steps: Number of forward steps to animate.
        step_delay_s: Delay between steps for visual readability.
    """
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=128, num_layers=2, num_heads=8)
    model.eval()

    with AttentionTracker(model, enable_logging=False) as tracker:
        stats = tracker.get_memory_stats()
        dashboard = RealtimeMemoryDashboard(refresh_interval_ms=120, max_points=120)
        figure, _animation = dashboard.create_animation(stats)

        plt.show(block=False)
        plt.pause(0.05)

        with torch.no_grad():
            for step in range(total_steps):
                seq_len = 12 + (step % 12)
                batch_size = 2 + (step % 2)
                input_ids = _build_input(batch_size, seq_len, vocab_size=1000)
                if input_ids is None:
                    # TODO: Add structured dropped-frame diagnostics.
                    continue

                model(input_ids)
                dashboard.update_from_stats(stats)

                plt.pause(0.001)
                time.sleep(step_delay_s)

        print_memory_summary(stats)
        plt.pause(0.8)
        plt.close(figure)


if __name__ == "__main__":
    run_realtime_demo()
