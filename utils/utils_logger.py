# codes/utils_logger.py
import os
import sys
import logging

def get_logger(name="ViR_MFS", log_file=None, log_level=logging.INFO):
    """Create a console/file logger.

    Args:
        name: Logger name.
        log_file: Optional file path for persistent logs.
        log_level: Python logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Clear old handlers to avoid duplicated log lines across repeated runs.
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(log_level)
    
    # Keep one compact format for both console and file logs.
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler.
    if log_file is not None:
        # Create the log directory before opening the file handler.
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
