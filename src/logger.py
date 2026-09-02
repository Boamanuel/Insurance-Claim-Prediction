"""
Shared logging configuration.

Every script in the pipeline calls get_logger(__name__) instead of
print(). This gives us:
  - timestamps and log levels on every line
  - a persistent log file per run (logs/pipeline.log) for audit/debugging
  - console output for interactive use
  - consistent formatting across modules

This is the main thing that separates a notebook (print() everywhere,
lost the moment the kernel restarts) from something you'd actually run
in production or hand to a colleague to debug at 2am.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times per name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. imported in multiple modules) — don't
        # duplicate handlers, which would duplicate every log line.
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — caps log file size instead of growing forever
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.LOG_DIR / config.LOG_FILE_NAME,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
