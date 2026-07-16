"""Shared validation primitives with no package-level dependencies."""

from __future__ import annotations

from typing import Any, Literal, overload


def validate_bool(value: Any, field_name: str) -> bool:
    """Validate a strict boolean control without accepting truthy substitutes."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be True or False")
    return value


@overload
def normalize_text(
    value: Any,
    field_name: str,
    *,
    required: Literal[True],
) -> str: ...


@overload
def normalize_text(
    value: Any,
    field_name: str,
    *,
    required: Literal[False] = False,
) -> str | None: ...


def normalize_text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
) -> str | None:
    """Normalize optional text and enforce one required/optional string contract."""
    if value is None:
        if required:
            raise ValueError(f"{field_name} must be a non-empty string")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized or None
