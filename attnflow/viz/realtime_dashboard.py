"""Real-time matplotlib dashboard for attention memory flow."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from attnflow.core.memory_stats import MemoryStats
from attnflow.utils.constants import DEFAULT_PLOT_STYLE


class RealtimeMemoryDashboard:
    """Render real-time memory timeline and KV cache growth in one view.

    The dashboard reads data from ``MemoryStats`` and updates visuals on each animation
    frame without changing the existing hook/tracker pipeline.

    TODO: Support WebSocket streaming backend for remote monitoring.
    TODO: Support Plotly/Gradio front-end rendering with the same data contract.
    """

    def __init__(
        self,
        refresh_interval_ms: int = 200,
        max_points: int = 200,
        style: str = DEFAULT_PLOT_STYLE,
    ) -> None:
        """Initialize dashboard settings.

        Args:
            refresh_interval_ms: Animation refresh interval in milliseconds.
            max_points: Maximum number of timeline points kept per layer.
            style: Matplotlib style name.
        """
        self.refresh_interval_ms = refresh_interval_ms
        self.max_points = max_points
        self.style = style

        self._fig: Optional[Figure] = None
        self._timeline_ax: Optional[Axes] = None
        self._growth_ax: Optional[Axes] = None
        self._lines: Dict[str, matplotlib.lines.Line2D] = {}
        self._total_bar = None
        self._status_text = None

    @staticmethod
    def _collect_layer_series(
        stats: MemoryStats,
        max_points: int,
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """Collect timeline series from all tracked layers.

        Args:
            stats: Memory statistics source.
            max_points: Maximum number of points per layer.

        Returns:
            Mapping of ``layer_name -> (timestamps, memory_mb)``.
        """
        series: Dict[str, Tuple[List[float], List[float]]] = {}
        for layer_name in stats.get_all_layers():
            timestamps, memory_values = stats.get_memory_timeline(layer_name)
            if not timestamps or not memory_values:
                continue

            if max_points > 0:
                timestamps = timestamps[-max_points:]
                memory_values = memory_values[-max_points:]

            series[layer_name] = (timestamps, memory_values)

        return series

    @staticmethod
    def _compute_current_total_mb(
        series: Dict[str, Tuple[List[float], List[float]]],
    ) -> float:
        """Compute current total KV memory from latest layer points.

        Args:
            series: Layer timeline mapping.

        Returns:
            Sum of latest memory value per layer in MB.
        """
        return sum(values[-1] for _, values in series.values() if values)

    @staticmethod
    def _compute_peak_total_mb(
        stats: MemoryStats,
        current_total_mb: float,
    ) -> float:
        """Compute peak axis upper bound for growth bar chart.

        Args:
            stats: Memory statistics source.
            current_total_mb: Current total memory in MB.

        Returns:
            Positive upper bound for x-axis.
        """
        summary = stats.get_summary()
        historical_peak = sum(
            float(layer_stats.get("peak_memory_mb", 0.0))
            for layer_stats in summary.values()
        )
        peak = max(historical_peak, current_total_mb)
        return peak if peak > 0.0 else 1.0

    def create_animation(
        self,
        stats: MemoryStats,
        title: str = "AttnFlow Realtime Attention Memory",
    ) -> Tuple[Figure, FuncAnimation]:
        """Create a live animation dashboard bound to ``MemoryStats``.

        Args:
            stats: Memory statistics source that keeps receiving snapshots.
            title: Figure title.

        Returns:
            Tuple of ``(figure, animation)`` for interactive display.
        """
        with plt.style.context(self.style):
            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(12, 7),
                gridspec_kw={"height_ratios": [3, 1]},
            )

        timeline_ax, growth_ax = axes
        fig.suptitle(title, fontsize=14, fontweight="bold")

        timeline_ax.set_xlabel("Time (seconds)")
        timeline_ax.set_ylabel("Memory (MB)")
        timeline_ax.set_title("Per-layer Memory Timeline")
        timeline_ax.grid(True, alpha=0.3)

        growth_ax.set_title("Global KV Cache Growth")
        growth_ax.set_xlabel("Memory (MB)")
        growth_ax.set_yticks([0])
        growth_ax.set_yticklabels(["Total KV"])

        self._fig = fig
        self._timeline_ax = timeline_ax
        self._growth_ax = growth_ax
        self._total_bar = growth_ax.barh([0], [0.0], color="tab:green", alpha=0.75)
        self._status_text = timeline_ax.text(
            0.5,
            0.5,
            "Waiting for attention snapshots...",
            ha="center",
            va="center",
            transform=timeline_ax.transAxes,
            fontsize=11,
            alpha=0.7,
        )

        self.update_from_stats(stats)

        animation = FuncAnimation(
            fig,
            lambda _frame: self.update_from_stats(stats),
            interval=self.refresh_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        animation._draw_was_started = True
        setattr(fig, "_attnflow_animation", animation)
        return fig, animation

    def update_from_stats(self, stats: MemoryStats) -> Sequence[matplotlib.artist.Artist]:
        """Refresh dashboard artists from the latest memory snapshots.

        Args:
            stats: Memory statistics source.

        Returns:
            Sequence of updated matplotlib artists.
        """
        if self._timeline_ax is None or self._growth_ax is None or self._total_bar is None:
            raise RuntimeError("Dashboard is not initialized. Call create_animation first.")

        series = self._collect_layer_series(stats, self.max_points)

        if not series:
            if self._status_text is not None:
                self._status_text.set_visible(True)
            self._total_bar[0].set_width(0.0)
            self._growth_ax.set_xlim(0.0, 1.0)
            return [self._total_bar[0]]

        if self._status_text is not None:
            self._status_text.set_visible(False)

        for layer_name, (timestamps, memory_values) in series.items():
            if layer_name in self._lines:
                line = self._lines[layer_name]
                line.set_data(timestamps, memory_values)
            else:
                line, = self._timeline_ax.plot(
                    timestamps,
                    memory_values,
                    marker="o",
                    linewidth=2,
                    label=layer_name,
                )
                self._lines[layer_name] = line

        for layer_name in list(self._lines.keys()):
            if layer_name not in series:
                line = self._lines.pop(layer_name)
                line.remove()

        all_times: List[float] = [t for times, _ in series.values() for t in times]
        all_memory: List[float] = [m for _, values in series.values() for m in values]

        if all_times and all_memory:
            time_min = min(all_times)
            time_max = max(all_times)
            if time_max <= time_min:
                time_max = time_min + 1e-6

            memory_max = max(all_memory)
            memory_upper = memory_max * 1.2 if memory_max > 0.0 else 1.0

            self._timeline_ax.set_xlim(time_min, time_max)
            self._timeline_ax.set_ylim(0.0, memory_upper)

        self._timeline_ax.legend(loc="upper left", fontsize=9)

        current_total_mb = self._compute_current_total_mb(series)
        peak_total_mb = self._compute_peak_total_mb(stats, current_total_mb)

        self._total_bar[0].set_width(current_total_mb)
        self._growth_ax.set_xlim(0.0, peak_total_mb * 1.1)

        self._growth_ax.set_title(
            f"Global KV Cache Growth  |  Current: {current_total_mb:.2f} MB",
            fontsize=11,
        )

        updated_artists: List[matplotlib.artist.Artist] = [self._total_bar[0]]
        updated_artists.extend(self._lines.values())
        return updated_artists
