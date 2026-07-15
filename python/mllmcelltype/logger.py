"""Logging module for LLMCellType."""

from __future__ import annotations

import datetime
import logging
import os

logger = logging.getLogger("llmcelltype")
if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
    logger.addHandler(logging.NullHandler())

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".mllmcelltype", "logs")

# Track initialization state for idempotent setup
_logging_initialized = False
_current_log_file: str | None = None
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level(log_level: str) -> tuple[str, int]:
    """Normalize a documented log level to its canonical name and value."""
    if not isinstance(log_level, str):
        raise ValueError(f"Invalid log level: {log_level!r}")
    level_name = log_level.strip().upper()
    level = LOG_LEVELS.get(level_name)
    if level is None:
        raise ValueError(
            f"Invalid log level: {log_level!r}. Must be one of: {', '.join(LOG_LEVELS)}"
        )
    return level_name, level


def _remove_file_handlers() -> None:
    """Close and detach every file handler owned by the package logger."""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def setup_logging(log_dir: str | None = None, log_level: str = "INFO") -> None:
    """Setup logging configuration.

    This function is idempotent - multiple calls with the same parameters
    will not create duplicate handlers. If called with different parameters,
    old handlers are cleaned up before adding new ones.

    Args:
        log_dir: Directory to store log files. If None, uses default directory.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    """
    global _logging_initialized, _current_log_file

    # Set log directory
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    _, level = _resolve_log_level(log_level)

    # If already initialized with the same directory, just update level
    if _logging_initialized:
        current_dir = os.path.dirname(_current_log_file) if _current_log_file else None
        if current_dir == log_dir:
            logger.setLevel(level)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(level)
            return
        # Different directory requested: clean up old handler and fall through
        _remove_file_handlers()

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llmcelltype_{timestamp}.log")

    # Remove any existing file handlers to prevent duplication
    _remove_file_handlers()

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)

    # Mark as initialized
    _logging_initialized = True
    _current_log_file = log_file

    logger.info(f"Logging initialized. Log file: {log_file}")


def write_log(message: str, level: str = "INFO") -> None:
    """Write a message to the log.

    Args:
        message: Message to log
        level: Log level (debug, info, warning, error, critical)

    """
    _, level_value = _resolve_log_level(level)
    logger.log(level_value, message)


def reset_logging() -> None:
    """Reset logging state to allow re-initialization.

    This is useful for testing or when you need to change the log directory
    after initial setup. It closes all file handlers and resets the
    initialization flag.
    """
    global _logging_initialized, _current_log_file

    # Close and remove all file handlers
    _remove_file_handlers()
    logger.setLevel(logging.NOTSET)

    # Reset state
    _logging_initialized = False
    _current_log_file = None


def get_current_log_file() -> str | None:
    """Get the path to the current log file.

    Returns:
        The path to the current log file, or None if logging is not initialized.
    """
    return _current_log_file
