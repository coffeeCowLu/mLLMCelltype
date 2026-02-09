"""MiniMax provider module for LLMCellType."""

from __future__ import annotations

import json
import re
import time

import requests

from ..logger import write_log
from ..url_utils import get_working_minimax_endpoint, validate_base_url


def process_minimax(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> list[str]:
    """Process request using MiniMax models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'minimax-m2.1', 'minimax-m2', 'MiniMax-Text-01')
        api_key: MiniMax API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting MiniMax API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "MiniMax API key is missing or empty"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Use custom URL or smart selection
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_working_minimax_endpoint(api_key)
        write_log(f"Using smart-selected endpoint: {url}")

    write_log(f"Using model: {model}")
    write_log(f"API URL: {url}")

    # Prepare the request body - use the same format as in R version
    body = {
        "model": model,
        "messages": [{"role": "user", "name": "user", "content": prompt}],
    }

    write_log("Sending API request...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            # Log request details for debugging
            write_log(f"Request URL: {url}", level="debug")
            write_log(f"Request headers: {headers}", level="debug")
            write_log(f"Request body: {json.dumps(body)}", level="debug")

            response = requests.post(url=url, headers=headers, data=json.dumps(body), timeout=30)

            # Log response details
            write_log(f"Response status code: {response.status_code}", level="debug")
            write_log(f"Response headers: {response.headers}", level="debug")

            # Check for errors
            if response.status_code != 200:
                try:
                    error_message = response.json()
                    write_log(f"MiniMax API request failed: {error_message}", level="error")
                    write_log(
                        f"Error details: {error_message.get('error', {}).get('message', 'Unknown error')}"
                    )
                except (ValueError, KeyError, json.JSONDecodeError):
                    write_log(
                        f"MiniMax API request failed with status {response.status_code}",
                        level="error",
                    )
                    write_log(f"Response text: {response.text}", level="debug")

                # If rate limited, wait and retry
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            # Parse the response
            content = response.json()

            # MiniMax returns HTTP 200 even for business errors (e.g. insufficient
            # balance).  The real status lives in base_resp.status_code (0 = OK).
            base_resp = content.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                error_msg = base_resp.get("status_msg", "Unknown MiniMax error")
                write_log(f"MiniMax API error: {error_msg}", level="error")
                raise ValueError(f"MiniMax API error: {error_msg}")

            # Parse response using the same format as in R version
            choices = content.get("choices")
            if (
                choices
                and len(choices) > 0
                and "message" in choices[0]
                and "content" in choices[0]["message"]
            ):
                response_content = choices[0]["message"]["content"]
                # Strip <think>...</think> reasoning block (MiniMax M2.1 Coding Plan)
                response_content = re.sub(
                    r"<think>[\s\S]*?</think>\s*", "", response_content
                )
                res = response_content.strip().split("\n")
            else:
                write_log(f"Unexpected response format: {content}")
                raise ValueError(f"Unexpected response format: {content}")

            write_log(f"Got response with {len(res)} lines")
            write_log(f"Raw response from MiniMax:\n{res}", level="debug")

            # Clean up results (remove commas at the end of lines)
            return [line.rstrip(",") for line in res]

        except Exception as e:
            # Non-retryable HTTP client errors — fail immediately
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response is not None
                and e.response.status_code < 500
            ):
                raise
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
