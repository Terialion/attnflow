"""CLI output utilities for memory statistics visualization."""

from typing import Dict, List
from attnflow.core.memory_stats import MemoryStats
from attnflow.utils.constants import (
    TABLE_SEPARATOR_WIDTH,
    LAYER_NAME_COLUMN_WIDTH,
    MEMORY_COLUMN_WIDTH,
    SEQ_LEN_COLUMN_WIDTH,
    TIMELINE_SEPARATOR_WIDTH,
    TIME_COLUMN_WIDTH,
)


def _print_separator(width: int) -> None:
    """Print a horizontal separator line."""
    print("=" * width)


def _print_divider(width: int) -> None:
    """Print a horizontal divider line."""
    print("-" * width)


def print_memory_summary(stats: MemoryStats) -> None:
    """
    Print memory summary for all layers in table format.
    
    Args:
        stats: MemoryStats instance containing collected data
    """
    summary = stats.get_summary()
    
    if not summary:
        print("No memory statistics available")
        return
    
    print("\n" + "=" * TABLE_SEPARATOR_WIDTH)
    print("ATTENTION MEMORY SUMMARY".center(TABLE_SEPARATOR_WIDTH))
    _print_separator(TABLE_SEPARATOR_WIDTH)
    
    # Header row
    layer_col = f"{'Layer Name':<{LAYER_NAME_COLUMN_WIDTH}}"
    memory_col = f"{'Peak Memory (MB)':<{MEMORY_COLUMN_WIDTH}}"
    seq_col = f"{'Max Seq Len':<{SEQ_LEN_COLUMN_WIDTH}}"
    print(f"{layer_col}{memory_col}{seq_col}")
    _print_divider(TABLE_SEPARATOR_WIDTH)
    
    # Data rows
    total_peak = 0.0
    for layer_name in sorted(summary.keys()):
        stats_dict = summary[layer_name]
        peak_mb = stats_dict["peak_memory_mb"]
        seq_len = stats_dict["peak_sequence_length"]
        
        layer_col = f"{layer_name:<{LAYER_NAME_COLUMN_WIDTH}}"
        memory_col = f"{peak_mb:<{MEMORY_COLUMN_WIDTH}.2f}"
        seq_col = f"{seq_len:<{SEQ_LEN_COLUMN_WIDTH}}"
        print(f"{layer_col}{memory_col}{seq_col}")
        total_peak += peak_mb
    
    _print_divider(TABLE_SEPARATOR_WIDTH)
    
    # Footer row
    total_col = f"{'TOTAL':<{LAYER_NAME_COLUMN_WIDTH}}"
    total_mem_col = f"{total_peak:<{MEMORY_COLUMN_WIDTH}.2f}"
    print(f"{total_col}{total_mem_col}")
    _print_separator(TABLE_SEPARATOR_WIDTH)
    print()


def print_memory_timeline(stats: MemoryStats, layer_name: str) -> None:
    """
    Print memory timeline for a specific layer.
    
    Args:
        stats: MemoryStats instance
        layer_name: Name of the layer to display
    """
    timestamps, memory_values = stats.get_memory_timeline(layer_name)
    
    if not timestamps:
        print(f"No data available for layer: {layer_name}")
        return
    
    print(f"\nMemory Timeline for {layer_name}")
    _print_divider(TIMELINE_SEPARATOR_WIDTH)
    
    # Header row
    time_col = f"{'Time (s)':<{TIME_COLUMN_WIDTH}}"
    memory_col = f"{'Memory (MB)':<{TIME_COLUMN_WIDTH}}"
    print(f"{time_col}{memory_col}")
    _print_divider(TIMELINE_SEPARATOR_WIDTH)
    
    # Data rows
    for t, mem in zip(timestamps, memory_values):
        time_col = f"{t:<{TIME_COLUMN_WIDTH}.4f}"
        memory_col = f"{mem:<{TIME_COLUMN_WIDTH}.2f}"
        print(f"{time_col}{memory_col}")
    
    _print_divider(TIMELINE_SEPARATOR_WIDTH)
    print()


def print_all_timelines(stats: MemoryStats) -> None:
    """
    Print memory timelines for all tracked layers.
    
    Args:
        stats: MemoryStats instance
    """
    for layer_name in stats.get_all_layers():
        print_memory_timeline(stats, layer_name)
