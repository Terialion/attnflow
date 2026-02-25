"""Unit tests for CLI output utilities."""

import pytest
from io import StringIO
import sys

from attnflow.core.memory_stats import MemoryStats
from attnflow.viz.cli_output import (
    print_memory_summary,
    print_memory_timeline,
    print_all_timelines,
)


class TestCliOutput:
    """Tests for CLI output functions."""
    
    def _capture_output(self, func, *args, **kwargs) -> str:
        """Capture stdout from a function."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            func(*args, **kwargs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output
    
    def test_print_memory_summary_empty(self) -> None:
        """Test printing summary with no data."""
        stats = MemoryStats()
        output = self._capture_output(print_memory_summary, stats)
        
        assert "No memory statistics available" in output
    
    def test_print_memory_summary_single_layer(self) -> None:
        """Test printing summary with single layer."""
        stats = MemoryStats()
        stats.record_snapshot("layer_0", 1024*1024, 1024*1024, 10)
        
        output = self._capture_output(print_memory_summary, stats)
        
        assert "ATTENTION MEMORY SUMMARY" in output
        assert "layer_0" in output
        assert "2.00" in output  # 2 MB
        assert "10" in output  # seq_len
        assert "TOTAL" in output
    
    def test_print_memory_summary_multiple_layers(self) -> None:
        """Test printing summary with multiple layers."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1024*1024, 1024*1024, 10)
        stats.record_snapshot("layer_1", 2*1024*1024, 2*1024*1024, 20)
        
        output = self._capture_output(print_memory_summary, stats)
        
        assert "layer_0" in output
        assert "layer_1" in output
        assert "4.00" in output  # total
    
    def test_print_memory_timeline_empty(self) -> None:
        """Test printing timeline with no data."""
        stats = MemoryStats()
        output = self._capture_output(print_memory_timeline, stats, "nonexistent")
        
        assert "No data available" in output
    
    def test_print_memory_timeline_single_layer(self) -> None:
        """Test printing timeline for single layer."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1024*1024, 1024*1024, 10)
        stats.record_snapshot("layer_0", 2*1024*1024, 2*1024*1024, 20)
        
        output = self._capture_output(print_memory_timeline, stats, "layer_0")
        
        assert "Memory Timeline for layer_0" in output
        assert "Time (s)" in output
        assert "Memory (MB)" in output
        # Should have 2 data rows
        lines = output.strip().split('\n')
        data_lines = [l for l in lines if l and not any(
            keyword in l for keyword in 
            ["Memory Timeline", "Time (s)", "Memory (MB)", "-", "=="]
        )]
        assert len(data_lines) >= 2
    
    def test_print_all_timelines(self) -> None:
        """Test printing timelines for all layers."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1024*1024, 1024*1024, 10)
        stats.record_snapshot("layer_1", 1024*1024, 1024*1024, 20)
        
        output = self._capture_output(print_all_timelines, stats)
        
        assert "Memory Timeline for layer_0" in output
        assert "Memory Timeline for layer_1" in output
    
    def test_summary_formatting(self) -> None:
        """Test that summary is properly formatted."""
        stats = MemoryStats()
        
        # Use values for easy verification
        stats.record_snapshot("test_layer_name_long", 1024*1024, 1024*1024, 100)
        
        output = self._capture_output(print_memory_summary, stats)
        
        # Check for proper column alignment
        lines = output.split('\n')
        
        # Should have header, separator, data, separator, total, separator
        assert len([l for l in lines if '=' in l]) >= 2
        assert len([l for l in lines if '-' in l]) >= 2
    
    def test_timeline_formatting(self) -> None:
        """Test that timeline is properly formatted."""
        stats = MemoryStats()
        
        stats.record_snapshot("layer_0", 1024*1024, 1024*1024, 10)
        
        output = self._capture_output(print_memory_timeline, stats, "layer_0")
        
        # Should contain header row and separator
        assert "Time (s)" in output
        assert "Memory (MB)" in output
        assert "-" * 50 in output or ("--" in output)
