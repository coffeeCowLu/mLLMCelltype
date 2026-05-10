"""Gemini provider module for LLMCellType."""

from __future__ import annotations

import time
from typing import Any

from ..logger import write_log
from .common import ensure_api_key


def _parse_gemini_response(response: Any) -> list[str]:
    """Parse Gemini SDK response into cleaned non-empty lines."""
    response_text = getattr(response, "text", None)
    if response_text is None:
        raise ValueError("Gemini response missing text content")
    if not isinstance(response_text, str):
        write_log("Unexpected non-string response content from Gemini, coercing to string", level="warning")
        response_text = str(response_text)

    return [
        line.strip().rstrip(",")
        for line in response_text.strip().split("\n")
        if line.strip().rstrip(",")
    ]


def process_gemini(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using Google Gemini models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite')
        api_key: Google API key
        base_url: Optional custom base URL (Note: Gemini uses SDK, base_url may not be applicable)

    Returns:
        List[str]: Processed responses, one per cluster

    Raises:
        ImportError: If google-genai package is not installed

    """
    # Lazy import - only load google-genai when actually using Gemini
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "Gemini provider requires 'google-genai' package. "
            "Install it with: pip install 'mllmcelltype[gemini]' or pip install google-genai"
        ) from e

    write_log(f"Starting Gemini API request with model: {model}")

    # Warn if base_url is provided (Gemini SDK doesn't support custom URLs)
    if isinstance(base_url, str) and base_url.strip():
        write_log(
            "base_url parameter is ignored for Gemini (SDK doesn't support custom URLs)",
            level="warning",
        )

    api_key = ensure_api_key(api_key, "Google")

    # Initialize the client
    client = genai.Client(api_key=api_key)
    write_log(f"Using model: {model}")

    # Set up retry parameters
    max_retries = 3
    retry_delay = 2

    # Try to generate content with retries
    for attempt in range(max_retries):
        try:
            write_log("Sending API request...")

            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4096),
            )

            result = _parse_gemini_response(response)
            write_log(f"Got response with {len(result)} lines")
            write_log(f"Raw response from Gemini:\n{result}", level="debug")
            return result

        except ValueError:
            # Response shape/content errors are deterministic; fail fast.
            raise
        except Exception as e:
            write_log(
                f"Error during API call (attempt {attempt + 1}/{max_retries}): {e!s}",
                level="error",
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...", level="warning")
                time.sleep(wait_time)
            else:
                raise
