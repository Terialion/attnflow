"""Logging utilities for AttnFlow."""

import logging
import sys

from attnflow.utils.constants import LOG_TIME_FORMAT, DEFAULT_LOG_LEVEL


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Logger instances are cached by Python's logging module, so multiple calls
    with the same name return the same logger object.
    
    Args:
        name: Logger name (typically module __name__)
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times to the same logger
    if logger.handlers:
        return logger
    
    # Configure logger level
    logger.setLevel(level)
    
    # Create stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter with consistent format
    formatter = logging.Formatter(
        fmt="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt=LOG_TIME_FORMAT
    )
    handler.setFormatter(formatter)
    
    # Attach handler to logger
    logger.addHandler(handler)
    
    return logger
