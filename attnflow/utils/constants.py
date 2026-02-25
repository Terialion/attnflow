"""Project-wide constants for AttnFlow.

Centralizes all magic numbers and configuration values to improve
maintainability and consistency across the codebase.
"""

# ============================================================================
# Memory Unit Conversions
# ============================================================================
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

# ============================================================================
# Attention Layer Configuration
# ============================================================================
# Default number of attention heads (used when not inferrable)
DEFAULT_NUM_HEADS = 8

# Default head dimension if num_heads is unknown
DEFAULT_HEAD_DIM = 64

# Default data type size in bytes (float32)
DEFAULT_DTYPE_BYTES = 4

# Keywords for identifying attention layers in model
ATTENTION_LAYER_KEYWORDS = (
    "attention",
    "self_attn",
    "cross_attn",
    "multihead",
    "self.attention",
)

# ============================================================================
# Visualization Configuration
# ============================================================================
DEFAULT_PLOT_STYLE = "default"
DEFAULT_DPI = 150

# Table formatting widths
TABLE_SEPARATOR_WIDTH = 70
LAYER_NAME_COLUMN_WIDTH = 30
MEMORY_COLUMN_WIDTH = 20
SEQ_LEN_COLUMN_WIDTH = 15

# Timeline printing widths
TIMELINE_SEPARATOR_WIDTH = 50
TIME_COLUMN_WIDTH = 15

# ============================================================================
# Logging
# ============================================================================
DEFAULT_LOG_LEVEL = "INFO"
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
