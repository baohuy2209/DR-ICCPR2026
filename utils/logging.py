"""Structured logging utilities for research experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "drdg", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger printing to stdout and optionally to a log file.

    Args:
        name: Name of the logger.
        log_file: Optional path to write log output.
        level: Logging level (e.g., logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
