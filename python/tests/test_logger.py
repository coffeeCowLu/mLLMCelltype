"""High-value tests for logging stability and error handling."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from mllmcelltype.logger import (
    get_current_log_file,
    logger,
    reset_logging,
    setup_logging,
    write_log,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Ensure logger state isolation across tests."""
    reset_logging()
    yield
    reset_logging()


def _file_handler_count() -> int:
    return sum(1 for handler in logger.handlers if isinstance(handler, logging.FileHandler))


def test_package_logger_has_non_configuring_default_handler():
    """Test library imports remain quiet without configuring application logging."""
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def test_setup_logging_idempotent_same_dir_updates_level(tmp_path: Path):
    """Test repeated setup on same directory keeps a single file handler."""
    setup_logging(log_dir=str(tmp_path), log_level="INFO")
    first_log_file = get_current_log_file()

    setup_logging(log_dir=str(tmp_path), log_level="DEBUG")
    second_log_file = get_current_log_file()

    assert first_log_file == second_log_file
    assert _file_handler_count() == 1
    assert logger.level == logging.DEBUG


def test_setup_logging_switch_directory_replaces_file_handler(tmp_path: Path):
    """Test changing log directory rotates file handler target safely."""
    dir_one = tmp_path / "logs_one"
    dir_two = tmp_path / "logs_two"

    setup_logging(log_dir=str(dir_one), log_level="INFO")
    first = get_current_log_file()

    setup_logging(log_dir=str(dir_two), log_level="INFO")
    second = get_current_log_file()

    assert first is not None
    assert second is not None
    assert first != second
    assert str(dir_two) in second
    assert _file_handler_count() == 1


def test_write_log_invalid_level_raises_value_error():
    """Test invalid write_log level fails with clear validation error."""
    with pytest.raises(ValueError, match="Invalid log level"):
        write_log("hello", level="INVALID")


def test_setup_logging_invalid_level_raises_value_error(tmp_path: Path):
    """Test setup_logging validates log level names."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(log_dir=str(tmp_path), log_level="NOPE")


def test_setup_and_write_log_share_level_normalization(tmp_path: Path):
    """Test setup and writes accept the same case-insensitive canonical levels."""
    setup_logging(log_dir=str(tmp_path), log_level=" debug ")

    with patch.object(logger, "log") as mock_log:
        write_log("hello", level=" warning ")

    assert logger.level == logging.DEBUG
    mock_log.assert_called_once_with(logging.WARNING, "hello")


def test_write_log_rejects_non_level_logger_attribute():
    """Test logger attributes cannot be mistaken for supported log levels."""
    with pytest.raises(ValueError, match="Invalid log level"):
        write_log("hello", level="handlers")


def test_write_log_rejects_non_string_level():
    """Test invalid level types fail with the same explicit contract."""
    with pytest.raises(ValueError, match="Invalid log level"):
        write_log("hello", level=10)  # type: ignore[arg-type]


def test_reset_logging_clears_current_log_file(tmp_path: Path):
    """Test reset removes handlers and restores the inherited logger level."""
    setup_logging(log_dir=str(tmp_path), log_level="INFO")
    assert get_current_log_file() is not None

    reset_logging()

    assert get_current_log_file() is None
    assert _file_handler_count() == 0
    assert logger.level == logging.NOTSET
