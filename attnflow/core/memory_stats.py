"""Memory statistics tracking for attention layers."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

from attnflow.utils.constants import BYTES_PER_MB


@dataclass
class LayerMemorySnapshot:
    """
    Snapshot of memory state for a single layer at a specific time.
    
    Attributes:
        layer_name: Name/identifier of the layer
        timestamp: Elapsed time in seconds since tracking started
        k_cache_size: Size of K cache in bytes
        v_cache_size: Size of V cache in bytes
        sequence_length: Current sequence length
    """
    layer_name: str
    timestamp: float
    k_cache_size: int
    v_cache_size: int
    sequence_length: int
    
    @property
    def total_memory(self) -> int:
        """Get total memory usage (K + V cache) in bytes."""
        return self.k_cache_size + self.v_cache_size
    
    @property
    def memory_mb(self) -> float:
        """Get memory usage in MB."""
        return self.total_memory / BYTES_PER_MB


class MemoryStats:
    """
    Track memory statistics across layers and time.
    
    Manages snapshots of KV cache sizes for all layers during forward pass.
    Provides methods to query memory usage patterns and sequences with caching.
    """
    
    def __init__(self):
        """Initialize memory statistics tracker."""
        self._snapshots: Dict[str, List[LayerMemorySnapshot]] = {}
        self._start_time = time.perf_counter()
        # Cache for computed summaries to avoid redundant calculations
        self._summary_cache: Dict = {}
        self._cache_valid = False
    
    def record_snapshot(
        self,
        layer_name: str,
        k_cache_size: int,
        v_cache_size: int,
        sequence_length: int
    ) -> None:
        """
        Record a memory snapshot for a layer.
        
        Args:
            layer_name: Identifier for the attention layer
            k_cache_size: Current K cache size in bytes
            v_cache_size: Current V cache size in bytes
            sequence_length: Current sequence length
        """
        if layer_name not in self._snapshots:
            self._snapshots[layer_name] = []

        timestamp = time.perf_counter() - self._start_time
        existing = self._snapshots[layer_name]
        if existing and timestamp <= existing[-1].timestamp:
            timestamp = existing[-1].timestamp + 1e-9
        
        snapshot = LayerMemorySnapshot(
            layer_name=layer_name,
            timestamp=timestamp,
            k_cache_size=k_cache_size,
            v_cache_size=v_cache_size,
            sequence_length=sequence_length
        )
        self._snapshots[layer_name].append(snapshot)
        # Invalidate cache when new data arrives
        self._cache_valid = False
    
    def get_snapshots(self, layer_name: str) -> List[LayerMemorySnapshot]:
        """
        Get all snapshots for a specific layer.
        
        Args:
            layer_name: Identifier of the layer
            
        Returns:
            List of memory snapshots for the layer
        """
        return self._snapshots.get(layer_name, [])
    
    def get_all_layers(self) -> List[str]:
        """
        Get list of all tracked layers.
        
        Returns:
            Sorted list of layer names
        """
        return sorted(self._snapshots.keys())
    
    def get_peak_memory(self, layer_name: str) -> int:
        """
        Get peak memory usage for a layer in bytes.
        
        Args:
            layer_name: Identifier of the layer
            
        Returns:
            Peak memory usage in bytes, or 0 if no data
        """
        snapshots = self._snapshots.get(layer_name, [])
        if not snapshots:
            return 0
        return max(snap.total_memory for snap in snapshots)
    
    def get_peak_memory_mb(self, layer_name: str) -> float:
        """
        Get peak memory usage for a layer in MB.
        
        Args:
            layer_name: Identifier of the layer
            
        Returns:
            Peak memory usage in MB
        """
        return self.get_peak_memory(layer_name) / BYTES_PER_MB
    
    def get_memory_timeline(self, layer_name: str) -> Tuple[List[float], List[float]]:
        """
        Get memory timeline for a layer.
        
        Args:
            layer_name: Identifier of the layer
            
        Returns:
            Tuple of (timestamps, memory_values) where:
                - timestamps: List of elapsed times in seconds
                - memory_values: List of memory usage in MB
        """
        snapshots = self._snapshots.get(layer_name, [])
        if not snapshots:
            return [], []
        
        timestamps = [snap.timestamp for snap in snapshots]
        memory_mb = [snap.memory_mb for snap in snapshots]
        return timestamps, memory_mb
    
    def get_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all layers.
        
        Uses cache to avoid redundant calculations.
        
        Returns:
            Dictionary with layer names as keys and memory stats as values
            Format: {
                'layer_name': {
                    'peak_memory_mb': float,
                    'peak_sequence_length': int,
                    'snapshot_count': int
                }
            }
        """
        if self._cache_valid:
            return self._summary_cache
        
        summary = {}
        for layer_name in self.get_all_layers():
            snapshots = self._snapshots[layer_name]
            if snapshots:
                peak_mem = max(snap.total_memory for snap in snapshots)
                peak_seq_len = max(snap.sequence_length for snap in snapshots)
                summary[layer_name] = {
                    "peak_memory_mb": peak_mem / BYTES_PER_MB,
                    "peak_sequence_length": peak_seq_len,
                    "snapshot_count": len(snapshots),
                }
        
        # Cache the result
        self._summary_cache = summary
        self._cache_valid = True
        return summary
    
    def clear(self) -> None:
        """Clear all recorded snapshots and invalidate cache."""
        self._snapshots.clear()
        self._summary_cache.clear()
        self._cache_valid = False

