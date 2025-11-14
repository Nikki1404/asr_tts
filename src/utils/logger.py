import logging
import sys
import os
from .config import ALL_CONFIG


def get_logger(name: str) -> logging.Logger:
    """
    Centralized logger used across the whole application.
    Ensures:
      - consistent formats
      - safe directory creation
      - no duplicate handlers
      - prevents server crashes due to missing log path
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers when file imported multiple times
    if logger.handlers:
        return logger

    # Resolve log file path from config
    log_path = ALL_CONFIG.get("PATH", {}).get("log_file_global", "app.log")

    # Ensure directory exists
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Handlers
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
