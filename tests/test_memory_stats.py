"""Unit tests for memory statistics tracking."""

import pytest
from attnflow.core.memory_stats import MemoryStats, LayerMemorySnapshot
from attnflow.utils.constants import BYTES_PER_MB


class TestLayerMemorySnapshot:
    """Tests for LayerMemorySnapshot dataclass."""
    
    def test_snapshot_creation(self) -> None:
        """Test creating a memory snapshot."""
        snapshot = LayerMemorySnapshot(
            layer_name="test_layer",
            timestamp=1.0,
            k_cache_size=1024,
            v_cache_size=1024,
            sequence_length=10
        )
        assert snapshot.layer_name == "test_layer"
        assert snapshot.timestamp == 1.0
        assert snapshot.k_cache_size == 1024
        assert snapshot.v_cache_size == 1024
    
    def test_total_memory_property(self) -> None:
        """Test total_memory property calculation."""
        snapshot = LayerMemorySnapshot(
            layer_name="layer",
            timestamp=0.0,
            k_cache_size=2000,
            v_cache_size=3000,
            sequence_length=1
        )
        assert snapshot.total_memory == 5000
    
    def test_memory_mb_property(self) -> None:
        """Test memory_mb property conversion."""
        snapshot = LayerMemorySnapshot(
            layer_name="layer",
            timestamp=0.0,
            k_cache_size=BYTES_PER_MB,
            v_cache_size=BYTES_PER_MB,
            sequence_length=1
        )
        assert snapshot.memory_mb == 2.0


class TestMemoryStats:
    """Tests for MemoryStats tracking class."""
    
    def test_initialization(self) -> None:
        """Test MemoryStats initialization."""
        stats = MemoryStats()
        assert stats.get_all_layers() == []
        assert stats.get_summary() == {}
    
    def test_record_snapshot(self) -> None:
        """Test recording a single snapshot."""
        stats = MemoryStats()
        stats.record_snapshot(
            layer_name="layer_0",
            k_cache_size=1000,
            v_cache_size=1000,
            sequence_length=10
        )
        
        assert "layer_0" in stats.get_all_layers()
        snapshots = stats.get_snapshots("layer_0")
        assert len(snapshots) == 1
        assert snapshots[0].layer_name == "layer_0"
    
    def test_record_multiple_snapshots(self) -> None:
        """Test recording multiple snapshots for the same layer."""
        stats = MemoryStats()
        
        for seq_len in [10, 20, 30]:
            stats.record_snapshot(
                layer_name="layer_0",
                k_cache_size=seq_len * 1000,
                v_cache_size=seq_len * 1000,
                sequence_length=seq_len
            )
        
        snapshots = stats.get_snapshots("layer_0")
        assert len(snapshots) == 3
        assert snapshots[0].sequence_length == 10
        assert snapshots[2].sequence_length == 30
    
    def test_get_peak_memory(self) -> None:
        """Test getting peak memory for a layer."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        stats.record_snapshot("layer_0", 3000, 3000, 30)
        stats.record_snapshot("layer_0", 2000, 2000, 20)
        
        peak = stats.get_peak_memory("layer_0")
        assert peak == 6000  # 3000 + 3000
    
    def test_get_peak_memory_mb(self) -> None:
        """Test getting peak memory in MB."""
        stats = MemoryStats()
        
        # Record 1 MB
        stats.record_snapshot("layer_0", BYTES_PER_MB, BYTES_PER_MB, 10)
        
        peak_mb = stats.get_peak_memory_mb("layer_0")
        assert peak_mb == 2.0
    
    def test_get_peak_memory_nonexistent_layer(self) -> None:
        """Test getting peak memory for nonexistent layer returns 0."""
        stats = MemoryStats()
        
        peak = stats.get_peak_memory("nonexistent")
        assert peak == 0
    
    def test_get_memory_timeline(self) -> None:
        """Test getting memory timeline."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        stats.record_snapshot("layer_0", 2000, 2000, 20)
        
        timestamps, memory_mb = stats.get_memory_timeline("layer_0")
        
        assert len(timestamps) == 2
        assert len(memory_mb) == 2
        # Timestamps should be increasing
        assert timestamps[0] < timestamps[1]
    
    def test_get_memory_timeline_empty(self) -> None:
        """Test getting timeline for nonexistent layer returns empty lists."""
        stats = MemoryStats()
        
        timestamps, memory_mb = stats.get_memory_timeline("nonexistent")
        
        assert timestamps == []
        assert memory_mb == []
    
    def test_get_all_layers(self) -> None:
        """Test getting all tracked layers."""
        stats = MemoryStats()
        
        layers = ["layer_1", "layer_0", "layer_2"]
        for layer in layers:
            stats.record_snapshot(layer, 1000, 1000, 10)
        
        all_layers = stats.get_all_layers()
        
        # Should be sorted
        assert all_layers == ["layer_0", "layer_1", "layer_2"]
    
    def test_get_summary(self) -> None:
        """Test getting summary statistics."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        stats.record_snapshot("layer_0", 2000, 2000, 20)
        
        summary = stats.get_summary()
        
        assert "layer_0" in summary
        layer_stats = summary["layer_0"]
        assert layer_stats["peak_memory_mb"] == 4000 / BYTES_PER_MB
        assert layer_stats["peak_sequence_length"] == 20
        assert layer_stats["snapshot_count"] == 2
    
    def test_cache_validity(self) -> None:
        """Test that summary cache is properly invalidated."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        summary1 = stats.get_summary()
        
        # Add new snapshot
        stats.record_snapshot("layer_0", 2000, 2000, 20)
        summary2 = stats.get_summary()
        
        # Summary should be updated
        assert summary1["layer_0"]["snapshot_count"] == 1
        assert summary2["layer_0"]["snapshot_count"] == 2
    
    def test_clear(self) -> None:
        """Test clearing statistics."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        assert len(stats.get_all_layers()) == 1
        
        stats.clear()
        
        assert stats.get_all_layers() == []
        assert stats.get_summary() == {}
    
    def test_multiple_layers(self) -> None:
        """Test tracking multiple layers independently."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1000, 1000, 10)
        stats.record_snapshot("layer_1", 2000, 2000, 20)
        stats.record_snapshot("layer_0", 3000, 3000, 30)
        
        assert stats.get_peak_memory("layer_0") == 6000
        assert stats.get_peak_memory("layer_1") == 4000
        
        snapshots_0 = stats.get_snapshots("layer_0")
        assert len(snapshots_0) == 2
        
        snapshots_1 = stats.get_snapshots("layer_1")
        assert len(snapshots_1) == 1
