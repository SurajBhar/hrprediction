#!/usr/bin/env python
"""
Logger module for the hotel-reservation-prediction project.

Provides a simple rotating-file logger suitable for small-scale experiments.
Logs are written to `logs/app.log` and rotated when the file reaches a size limit.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Configuration constants
LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FILE = LOGS_DIR / "mlapp.log"
MAX_BYTES = 5 * 1024 * 1024         # 5 MB per log file
BACKUP_COUNT = 3                    # keep up to 3 backup files


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a logger with a rotating file handler and console output.

    This logger will write messages to a file, rotating the file when it
    exceeds MAX_BYTES. A limited number of backup files are kept. Logs
    also appear on the console (stderr) for immediate feedback.

    Args:
        name (str): Name of the logger (e.g., __name__ of the calling module).
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Attach handlers only once
    if not logger.handlers:
        # File handler with size-based rotation
        file_handler = RotatingFileHandler(
            filename=DEFAULT_LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )

        # Console handler for stdout/stderr
        console_handler = logging.StreamHandler()

        # Common log message format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Set handler levels
        file_handler.setLevel(level)
        console_handler.setLevel(level)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
