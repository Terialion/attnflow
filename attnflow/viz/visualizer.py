"""Matplotlib-based visualizer for attention memory flow."""

from typing import Any, Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib

from attnflow.core.memory_stats import MemoryStats
from attnflow.utils.constants import DEFAULT_PLOT_STYLE, DEFAULT_DPI


class Visualizer:
    """
    Create visualizations of attention memory statistics.
    
    Generates plots showing KV cache memory growth over time
    and comparison across layers.
    
    TODO: Support interactive visualization with plotly
    TODO: Add heatmap visualization for all layers
    """
    
    def __init__(self, style: str = DEFAULT_PLOT_STYLE) -> None:
        """
        Initialize visualizer with matplotlib backend and style.
        
        Args:
            style: Matplotlib style name (default, seaborn, ggplot, etc.)
        """
        self.style = style
        # Use non-interactive backend for server environments
        matplotlib.use("Agg")
    
    def plot_memory_timeline(
        self,
        stats: MemoryStats,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot memory timeline for one or all layers.
        
        Args:
            stats: MemoryStats instance with collected data
            layer_name: Specific layer to plot, or None for all layers
            save_path: Path to save figure, or None to display only
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if layer_name:
                # Plot single layer
                self._plot_single_layer_timeline(ax, stats, layer_name)
            else:
                # Plot all layers
                self._plot_all_layers_timeline(ax, stats)
            
            # Styling
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_ylabel("Memory Usage (MB)", fontsize=12)
            ax.set_title("Attention Memory Timeline", fontsize=14, fontweight="bold")
            ax.legend(loc="upper left", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            
            return fig
    
    def plot_peak_memory_comparison(
        self,
        stats: MemoryStats,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot peak memory usage across layers.
        
        Args:
            stats: MemoryStats instance with collected data
            save_path: Path to save figure, or None to display only
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            layers = stats.get_all_layers()
            peak_memories = [stats.get_peak_memory_mb(layer) for layer in layers]
            
            # Create bar chart
            bars = ax.bar(range(len(layers)), peak_memories, color="steelblue", alpha=0.8)
            
            # Styling
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha="right")
            ax.set_ylabel("Peak Memory Usage (MB)", fontsize=12)
            ax.set_title("Peak Memory by Layer", fontsize=14, fontweight="bold")
            
            # Add value labels on bars
            self._add_bar_value_labels(ax, bars, peak_memories)
            
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            
            return fig
    
    @staticmethod
    def _plot_single_layer_timeline(
        ax: plt.Axes,
        stats: MemoryStats,
        layer_name: str
    ) -> None:
        """
        Plot memory timeline for a single layer.
        
        Args:
            ax: Matplotlib axes to plot on
            stats: MemoryStats instance
            layer_name: Name of the layer to plot
        """
        timestamps, memory_mb = stats.get_memory_timeline(layer_name)
        if timestamps:
            ax.plot(timestamps, memory_mb, marker="o", linewidth=2, label=layer_name)
    
    @staticmethod
    def _plot_all_layers_timeline(
        ax: plt.Axes,
        stats: MemoryStats
    ) -> None:
        """
        Plot memory timeline for all layers.
        
        Args:
            ax: Matplotlib axes to plot on
            stats: MemoryStats instance
        """
        for layer_name in stats.get_all_layers():
            timestamps, memory_mb = stats.get_memory_timeline(layer_name)
            if timestamps:
                ax.plot(timestamps, memory_mb, marker="o", label=layer_name, alpha=0.7)
    
    @staticmethod
    def _add_bar_value_labels(
        ax: plt.Axes,
        bars: Sequence[Any],
        values: Sequence[float],
    ) -> None:
        """
        Add value labels on top of bar chart.
        
        Args:
            ax: Matplotlib axes
            bars: Bar container from ax.bar()
            values: List of values for each bar
        """
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    
    @staticmethod
    def show(fig: plt.Figure) -> None:
        """
        Display figure.
        
        Args:
            fig: matplotlib Figure to display
        """
        plt.show()
