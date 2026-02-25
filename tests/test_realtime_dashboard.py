"""Unit tests for realtime memory dashboard visualization."""

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation

from attnflow.core.memory_stats import MemoryStats
from attnflow.viz.realtime_dashboard import RealtimeMemoryDashboard


pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything"
)


class TestRealtimeMemoryDashboard:
    """Tests for realtime dashboard rendering and edge handling."""

    def test_collect_layer_series_empty(self) -> None:
        """Collecting series from empty stats should return empty mapping."""
        stats = MemoryStats()

        series = RealtimeMemoryDashboard._collect_layer_series(stats, max_points=100)

        assert series == {}

    def test_update_from_stats_no_layers(self) -> None:
        """Dashboard should remain stable when no attention layer snapshots exist."""
        stats = MemoryStats()
        dashboard = RealtimeMemoryDashboard(refresh_interval_ms=100)
        _, _animation = dashboard.create_animation(stats)

        artists = dashboard.update_from_stats(stats)

        assert len(artists) >= 1
        assert dashboard._status_text is not None
        assert dashboard._status_text.get_visible()
        assert dashboard._total_bar is not None
        assert dashboard._total_bar[0].get_width() == 0.0

    def test_update_from_stats_single_layer(self) -> None:
        """Dashboard should draw a single-layer timeline and non-zero growth bar."""
        stats = MemoryStats()
        stats.record_snapshot("layer_0.self_attention", 1024 * 1024, 1024 * 1024, 8)
        stats.record_snapshot("layer_0.self_attention", 2 * 1024 * 1024, 2 * 1024 * 1024, 16)

        dashboard = RealtimeMemoryDashboard(refresh_interval_ms=100)
        _, _animation = dashboard.create_animation(stats)
        dashboard.update_from_stats(stats)

        assert "layer_0.self_attention" in dashboard._lines
        assert dashboard._total_bar is not None
        assert dashboard._total_bar[0].get_width() > 0.0

    def test_create_animation_returns_objects(self) -> None:
        """Animation builder should return matplotlib figure and FuncAnimation."""
        stats = MemoryStats()
        dashboard = RealtimeMemoryDashboard(refresh_interval_ms=100)

        fig, animation = dashboard.create_animation(stats)

        assert fig is not None
        assert isinstance(animation, FuncAnimation)
