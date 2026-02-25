"""
AttnFlow: Real-time Transformer Attention Memory Flow Visualization

A lightweight toolkit for visualizing and analyzing KV cache memory consumption
in Transformer models during inference.
"""

__version__ = "0.1.0"
__author__ = "AttnFlow Team"

from attnflow.core.tracker import AttentionTracker

__all__ = ["AttentionTracker"]
