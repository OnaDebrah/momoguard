import logging
import sys

def setup_logger(name="MoMoGuard"):
    """Configure root logger with console output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

logger = setup_logger()