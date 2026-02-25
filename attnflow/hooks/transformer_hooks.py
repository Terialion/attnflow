"""Transformer forward hooks for attention monitoring."""

from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn as nn

from attnflow.core.memory_stats import MemoryStats
from attnflow.utils.logger import get_logger
from attnflow.utils.constants import (
    ATTENTION_LAYER_KEYWORDS,
    DEFAULT_NUM_HEADS,
    DEFAULT_DTYPE_BYTES,
    BYTES_PER_MB,
)


logger = get_logger(__name__)


class TransformerHookManager:
    """
    Manages forward hooks for Transformer attention layers.
    
    Registers hooks on attention modules to capture KV cache sizes
    and tensor shapes during forward pass.
    
    TODO: Support multi-headed attention pattern matching
    TODO: Support different attention implementations (flash attention, etc.)
    """
    
    def __init__(self, model: nn.Module, memory_stats: MemoryStats) -> None:
        """
        Initialize hook manager.
        
        Args:
            model: PyTorch model to attach hooks to
            memory_stats: MemoryStats instance for recording data
        """
        self.model = model
        self.memory_stats = memory_stats
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    
    def register_hooks(self) -> int:
        """
        Register forward hooks on all attention layers.
        
        Identifies and hooks into attention modules by pattern matching
        common layer names like 'attention', 'self_attn', etc.
        
        Returns:
            Number of hooks registered
        """
        if self._hook_handles:
            logger.warning("Hooks already registered on this manager instance, skipping")
            return len(self._hook_handles)

        hook_count = 0
        
        for name, module in self.model.named_modules():
            if self._is_attention_layer(name):
                hook_count += 1
                handle = module.register_forward_hook(self._create_hook(name))
                self._hook_handles.append(handle)
                logger.debug(f"Registered hook on layer: {name}")
        
        logger.info(f"Successfully registered {hook_count} hooks")
        return hook_count
    
    def unregister_hooks(self) -> None:
        """Remove all registered hooks from model."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        logger.info("All hooks unregistered")
    
    def _is_attention_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer name matches attention layer patterns.
        
        Args:
            layer_name: Module name from model.named_modules()
            
        Returns:
            True if layer_name contains any attention keyword
        """
        if not layer_name:
            return False

        module_name = layer_name.lower().split(".")[-1]
        explicit_names = {
            "attention",
            "self_attention",
            "self_attn",
            "cross_attn",
        }

        if module_name in explicit_names:
            return True

        if module_name.startswith("multiheadattention") or module_name == "multihead":
            return True

        return module_name in ATTENTION_LAYER_KEYWORDS
    
    @staticmethod
    def _extract_tensor_shape(output: Any) -> Optional[Tuple[int, int, int]]:
        """
        Extract batch_size, seq_len, hidden_dim from module output.
        
        Handles both tensor and tuple outputs.
        
        Args:
            output: Module output (tensor, tuple, or other)
            
        Returns:
            Tuple of (batch_size, seq_len, hidden_dim) or None if extraction fails
        """
        try:
            if isinstance(output, torch.Tensor):
                if output.dim() >= 3:
                    return output.shape[:3]
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    tensor = output[0]
                    if tensor.dim() >= 3:
                        return tensor.shape[:3]
        except (AttributeError, IndexError, TypeError):
            pass
        return None
    
    @staticmethod
    def _estimate_cache_size(
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int = DEFAULT_NUM_HEADS,
        dtype_bytes: int = DEFAULT_DTYPE_BYTES,
    ) -> Tuple[int, int]:
        """
        Estimate K and V cache sizes in bytes.
        
        Formula: batch * seq_len * hidden_dim * dtype_size
        Both K and V caches have the same size in standard attention.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads (for reference)
            dtype_bytes: Data type size in bytes
            
        Returns:
            Tuple of (k_cache_size, v_cache_size) in bytes
        """
        cache_size = batch_size * seq_len * hidden_dim * dtype_bytes
        return cache_size, cache_size
    
    def _create_hook(
        self,
        layer_name: str,
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
        """
        Create a forward hook function for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Hook function that records memory statistics
        """
        def hook(module: nn.Module, input: Tuple[Any, ...], output: Any) -> None:
            """
            Forward hook implementation.
            
            Captures attention output shapes and estimates KV cache sizes.
            Called automatically by PyTorch when module completes forward pass.
            
            Args:
                module: The module being hooked
                input: Input tensors (unused)
                output: Output from the module
            """
            try:
                # Extract tensor dimensions
                shape = self._extract_tensor_shape(output)
                if shape is None:
                    return
                
                batch_size, seq_len, hidden_dim = shape
                
                # Estimate KV cache sizes
                k_cache_size, v_cache_size = self._estimate_cache_size(
                    batch_size, seq_len, hidden_dim
                )
                
                # Record snapshot
                self.memory_stats.record_snapshot(
                    layer_name=layer_name,
                    k_cache_size=k_cache_size,
                    v_cache_size=v_cache_size,
                    sequence_length=seq_len
                )
                
                # Log for debugging (only at debug level for performance)
                memory_mb = (k_cache_size + v_cache_size) / BYTES_PER_MB
                logger.debug(
                    f"[{layer_name}] seq_len={seq_len}, "
                    f"total_cache={memory_mb:.2f}MB"
                )
            
            except Exception as e:
                logger.warning(f"Error in hook for {layer_name}: {e}", exc_info=False)
        
        return hook
