"""Main attention tracking orchestrator."""

from typing import Optional
import torch.nn as nn

from attnflow.core.memory_stats import MemoryStats
from attnflow.hooks.transformer_hooks import TransformerHookManager
from attnflow.utils.logger import get_logger


logger = get_logger(__name__)


class AttentionTracker:
    """
    Main orchestrator for tracking attention memory flow.
    
    Manages hook registration, memory statistics collection, and provides
    unified interface for monitoring Transformer attention layers during forward pass.
    
    Can be used as a context manager for automatic cleanup:
    >>> with AttentionTracker(model) as tracker:
    ...     output = model(input_ids)
    ...     stats = tracker.get_memory_stats()
    """
    
    def __init__(self, model: nn.Module, enable_logging: bool = True):
        """
        Initialize attention tracker.
        
        Args:
            model: PyTorch model containing attention layers
            enable_logging: Whether to enable debug logging
        """
        self.model = model
        self.enable_logging = enable_logging
        
        self._memory_stats = MemoryStats()
        self._hook_manager = TransformerHookManager(model, self._memory_stats)
        self._hooks_registered = False
        self._is_tracking = False
        
        self._log_info(f"Initialized AttentionTracker for model: {model.__class__.__name__}")
    
    def register_hooks(self) -> None:
        """
        Register forward hooks on all attention layers.
        
        This method must be called before tracking begins.
        Hooks will capture attention tensor information during forward pass.
        
        Automatically handles duplicate registrations by skipping if already registered.
        
        TODO: Support custom layer patterns matching
        """
        if self._hooks_registered:
            self._log_warning("Hooks already registered, skipping...")
            return
        
        self._hook_manager.register_hooks()
        self._hooks_registered = True
        self._log_info("Hooks registered successfully")
    
    def unregister_hooks(self) -> None:
        """
        Unregister all forward hooks from the model.
        
        Should be called when tracking is no longer needed to prevent
        overhead from hook execution.
        """
        if not self._hooks_registered:
            return
        
        self._hook_manager.unregister_hooks()
        self._hooks_registered = False
        self._log_info("Hooks unregistered")
    
    def reset_stats(self) -> None:
        """
        Clear all collected memory statistics.
        
        Useful when running multiple tracking sessions with the same tracker.
        """
        self._memory_stats.clear()
        self._log_info("Statistics reset")
    
    def start_tracking(self) -> None:
        """
        Start collecting memory statistics.
        
        Must be called after register_hooks() and before model evaluation.
        
        Raises:
            RuntimeError: If hooks have not been registered yet
        """
        if not self._hooks_registered:
            raise RuntimeError("Hooks must be registered before tracking")
        
        self._is_tracking = True
        self._log_info("Tracking started")
    
    def stop_tracking(self) -> None:
        """Stop collecting memory statistics."""
        self._is_tracking = False
        self._log_info("Tracking stopped")
    
    def is_tracking(self) -> bool:
        """
        Check if currently tracking memory statistics.
        
        Returns:
            True if tracking is active, False otherwise
        """
        return self._is_tracking
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get collected memory statistics.
        
        Returns:
            MemoryStats object containing all recorded snapshots
        """
        return self._memory_stats
    
    def get_summary(self) -> dict:
        """
        Get summary of all memory statistics.
        
        Returns:
            Dictionary with layer-wise memory summary
            Format: {layer_name: {peak_memory_mb, peak_sequence_length, snapshot_count}}
        """
        return self._memory_stats.get_summary()
    
    def _log_info(self, message: str) -> None:
        """Log info message if logging is enabled."""
        if self.enable_logging:
            logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message if logging is enabled."""
        if self.enable_logging:
            logger.warning(message)
    
    def __enter__(self):
        """
        Context manager entry.
        
        Automatically registers hooks and starts tracking when entering the context.
        """
        self.register_hooks()
        self.start_tracking()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Automatically stops tracking and unregisters hooks when exiting the context.
        """
        self.stop_tracking()
        self.unregister_hooks()
