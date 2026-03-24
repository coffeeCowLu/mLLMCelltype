"""Tests for internal cache maintenance CLI behavior."""

from __future__ import annotations

from unittest.mock import patch

from mllmcelltype.cache_manager import clear_cache_cli, main


@patch("mllmcelltype.cache_manager.clear_cache")
def test_cache_manager_clear_flag(mock_clear_cache, capsys):
    """Test --clear clears cache and reports count."""
    mock_clear_cache.return_value = 5

    with patch("sys.argv", ["cache_manager", "--clear"]):
        exit_code = clear_cache_cli()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Cleared 5 cache files." in out


@patch("mllmcelltype.cache_manager.get_cache_stats")
def test_cache_manager_no_args_prints_info_and_tip(mock_get_cache_stats, capsys):
    """Test no-arg mode is non-interactive and prints cache status."""
    mock_get_cache_stats.return_value = {
        "exists": True,
        "path": "/tmp/cache",
        "count": 3,
        "size_mb": 1.25,
    }

    with patch("sys.argv", ["cache_manager"]):
        exit_code = clear_cache_cli()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Cache directory: /tmp/cache" in out
    assert "Number of cache files: 3" in out
    assert "Total cache size: 1.25 MB" in out
    assert "Tip: use --clear to remove all cache files." in out


@patch("mllmcelltype.cache_manager.get_cache_stats")
def test_cache_manager_info_flag_prints_only_info(mock_get_cache_stats, capsys):
    """Test --info prints cache stats without no-arg tip text."""
    mock_get_cache_stats.return_value = {
        "exists": True,
        "path": "/tmp/cache",
        "count": 2,
        "size_mb": 0.5,
    }

    exit_code = main(["--info"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Cache directory: /tmp/cache" in out
    assert "Number of cache files: 2" in out
    assert "Total cache size: 0.50 MB" in out
    assert "Tip: use --clear to remove all cache files." not in out
